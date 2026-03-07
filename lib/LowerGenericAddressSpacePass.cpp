// Copyright 2025 The Clspv Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Lowers the Generic address space (4) to Global (1).
// HIP/chipStar uses generic pointers extensively but clspv does not support
// the SPIR-V Generic storage class. This pass converts:
// - addrspacecast to/from AS4 into casts to/from AS1
// - inttoptr to AS4 into inttoptr to AS1
// - Removes no-op addrspacecasts (AS1 to AS1 after conversion)

#include "LowerGenericAddressSpacePass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "clspv/AddressSpace.h"

using namespace llvm;

static const unsigned kGenericAS = 4; // SPIR-V Generic / OpenCL Generic

// Change a pointer type's address space.
static Type *changePointerAS(Type *Ty, unsigned NewAS) {
  if (auto *PTy = dyn_cast<PointerType>(Ty))
    return PointerType::get(Ty->getContext(), NewAS);
  return Ty;
}

PreservedAnalyses
clspv::LowerGenericAddressSpacePass::run(Module &M,
                                         ModuleAnalysisManager &) {
  bool Changed = false;

  for (auto &F : M) {
    SmallVector<Instruction *, 16> ToErase;

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
          unsigned SrcAS = ASC->getSrcAddressSpace();
          unsigned DstAS = ASC->getDestAddressSpace();

          // Convert casts involving Generic AS
          if (SrcAS == kGenericAS || DstAS == kGenericAS) {
            unsigned NewSrcAS =
                (SrcAS == kGenericAS) ? clspv::AddressSpace::Global : SrcAS;
            unsigned NewDstAS =
                (DstAS == kGenericAS) ? clspv::AddressSpace::Global : DstAS;

            if (NewSrcAS == NewDstAS) {
              // No-op cast: just replace with source operand
              ASC->replaceAllUsesWith(ASC->getPointerOperand());
              ToErase.push_back(ASC);
            } else {
              // Different address spaces: create new cast
              auto *NewASC = new AddrSpaceCastInst(
                  ASC->getPointerOperand(),
                  PointerType::get(M.getContext(), NewDstAS), "",
                  ASC->getIterator());
              ASC->replaceAllUsesWith(NewASC);
              ToErase.push_back(ASC);
            }
            Changed = true;
          }
        } else if (auto *ITP = dyn_cast<IntToPtrInst>(&I)) {
          if (ITP->getType()->getPointerAddressSpace() == kGenericAS) {
            auto *NewITP = new IntToPtrInst(
                ITP->getOperand(0),
                PointerType::get(M.getContext(), clspv::AddressSpace::Global),
                "", ITP->getIterator());
            ITP->replaceAllUsesWith(NewITP);
            ToErase.push_back(ITP);
            Changed = true;
          }
        } else if (auto *PTI = dyn_cast<PtrToIntInst>(&I)) {
          // PtrToInt from Generic: operand should already be converted,
          // but check if the source is still AS4
          if (PTI->getPointerAddressSpace() == kGenericAS) {
            // The operand's address space will be updated when its
            // defining instruction is processed, but if it's a function
            // argument we need to handle it differently.
            // For now, just leave it — the addrspacecast that produces
            // the AS4 pointer will be converted above.
          }
        }
      }
    }

    for (auto *I : ToErase)
      I->eraseFromParent();
  }

  // Also handle constant expressions with addrspacecast to/from Generic
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (unsigned OpIdx = 0; OpIdx < I.getNumOperands(); ++OpIdx) {
          auto *CE = dyn_cast<ConstantExpr>(I.getOperand(OpIdx));
          if (!CE || CE->getOpcode() != Instruction::AddrSpaceCast)
            continue;

          auto *SrcTy = CE->getOperand(0)->getType();
          auto *DstTy = CE->getType();
          unsigned SrcAS = SrcTy->getPointerAddressSpace();
          unsigned DstAS = DstTy->getPointerAddressSpace();

          if (SrcAS != kGenericAS && DstAS != kGenericAS)
            continue;

          unsigned NewDstAS =
              (DstAS == kGenericAS) ? clspv::AddressSpace::Global : DstAS;

          if (SrcAS == NewDstAS) {
            // No-op: replace with source
            I.setOperand(OpIdx, CE->getOperand(0));
          } else {
            auto *NewCE = ConstantExpr::getAddrSpaceCast(
                CE->getOperand(0),
                PointerType::get(M.getContext(), NewDstAS));
            I.setOperand(OpIdx, NewCE);
          }
          Changed = true;
        }
      }
    }
  }

  // Mark __chip_var_* and __chip_module_has_no_IGBAs globals as
  // externally_initialized so that SPIRVProducerPass promotes them to
  // StorageBuffer SSBOs instead of PhysicalStorageBuffer OpVariables
  // (which cannot have initializers).
  // The externally_initialized attribute is lost during the SPIR-V
  // roundtrip (llvm-spirv -> SPIR-V -> llvm-spirv -r), so we restore it here.
  for (auto &GV : M.globals()) {
    if (GV.hasName() &&
        (GV.getName().starts_with("__chip_var_") ||
         GV.getName() == "__chip_module_has_no_IGBAs") &&
        GV.getType()->getAddressSpace() == clspv::AddressSpace::Global &&
        !GV.isExternallyInitialized()) {
      GV.setExternallyInitialized(true);
      Changed = true;
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
