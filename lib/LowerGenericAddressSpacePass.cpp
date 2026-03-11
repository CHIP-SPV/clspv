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
// the SPIR-V Generic storage class. This pass rewrites all AS4 pointer types
// to AS1 by rebuilding instructions that produce or consume AS4 pointers.

#include "LowerGenericAddressSpacePass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "clspv/AddressSpace.h"

using namespace llvm;

static const unsigned kGenericAS = 4; // SPIR-V Generic / OpenCL Generic
static const unsigned kGlobalAS = clspv::AddressSpace::Global;

// Map AS4 → AS1, leave everything else untouched.
static unsigned mapAS(unsigned AS) {
  return (AS == kGenericAS) ? kGlobalAS : AS;
}

// Return a pointer type with AS4 replaced by AS1.
static Type *remapPointerType(Type *Ty, LLVMContext &Ctx) {
  if (auto *PT = dyn_cast<PointerType>(Ty)) {
    unsigned AS = PT->getAddressSpace();
    if (AS == kGenericAS)
      return PointerType::get(Ctx, kGlobalAS);
  }
  return Ty;
}

PreservedAnalyses
clspv::LowerGenericAddressSpacePass::run(Module &M,
                                         ModuleAnalysisManager &) {
  bool Changed = false;

  // Iteratively rewrite instructions until no AS4 pointers remain.
  // Multiple iterations handle cases where rewriting one instruction
  // exposes new AS4 uses (e.g., GEP result type follows input AS).
  bool Progress = true;
  while (Progress) {
    Progress = false;

    for (auto &F : M) {
      SmallVector<Instruction *, 16> ToErase;

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
            unsigned SrcAS = ASC->getSrcAddressSpace();
            unsigned DstAS = ASC->getDestAddressSpace();

            if (SrcAS == kGenericAS || DstAS == kGenericAS) {
              unsigned NewSrcAS = mapAS(SrcAS);
              unsigned NewDstAS = mapAS(DstAS);

              if (NewSrcAS == NewDstAS) {
                ASC->replaceAllUsesWith(ASC->getPointerOperand());
                ToErase.push_back(ASC);
              } else {
                auto *NewASC = new AddrSpaceCastInst(
                    ASC->getPointerOperand(),
                    PointerType::get(M.getContext(), NewDstAS), "",
                    ASC->getIterator());
                ASC->replaceAllUsesWith(NewASC);
                ToErase.push_back(ASC);
              }
              Progress = true;
            }
          } else if (auto *ITP = dyn_cast<IntToPtrInst>(&I)) {
            if (ITP->getType()->getPointerAddressSpace() == kGenericAS) {
              // Create new inttoptr to AS1
              auto *NewITP = new IntToPtrInst(
                  ITP->getOperand(0),
                  PointerType::get(M.getContext(), kGlobalAS), "",
                  ITP->getIterator());
              // Cannot RAUW across address spaces — rewrite each user
              SmallVector<Use *, 8> Uses;
              for (auto &U : ITP->uses())
                Uses.push_back(&U);
              for (auto *U : Uses)
                U->set(NewITP);
              ToErase.push_back(ITP);
              Progress = true;
            }
          } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
            // GEP result type follows the pointer operand's AS.
            // If the pointer is AS4, rebuild with AS1 pointer.
            if (GEP->getType()->getPointerAddressSpace() == kGenericAS &&
                GEP->getPointerOperand()->getType()->getPointerAddressSpace() !=
                    kGenericAS) {
              // Pointer operand already remapped, but GEP type still shows
              // AS4. This shouldn't happen with opaque pointers — GEP result
              // AS follows input. Skip.
            }
          } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
            if (PHI->getType()->isPointerTy() &&
                PHI->getType()->getPointerAddressSpace() == kGenericAS) {
              // Check if all incoming values are now AS1
              bool AllAS1 = true;
              for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
                auto *V = PHI->getIncomingValue(i);
                if (V->getType()->isPointerTy() &&
                    V->getType()->getPointerAddressSpace() == kGenericAS) {
                  AllAS1 = false;
                  break;
                }
              }
              if (AllAS1) {
                // Create new PHI with AS1 type
                auto *NewPHI = PHINode::Create(
                    PointerType::get(M.getContext(), kGlobalAS),
                    PHI->getNumIncomingValues(), "", PHI->getIterator());
                for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i)
                  NewPHI->addIncoming(PHI->getIncomingValue(i),
                                      PHI->getIncomingBlock(i));
                SmallVector<Use *, 8> Uses;
                for (auto &U : PHI->uses())
                  Uses.push_back(&U);
                for (auto *U : Uses)
                  U->set(NewPHI);
                ToErase.push_back(PHI);
                Progress = true;
              }
            }
          } else if (auto *SI = dyn_cast<SelectInst>(&I)) {
            if (SI->getType()->isPointerTy() &&
                SI->getType()->getPointerAddressSpace() == kGenericAS) {
              auto *TV = SI->getTrueValue();
              auto *FV = SI->getFalseValue();
              bool TVAS1 = !TV->getType()->isPointerTy() ||
                           TV->getType()->getPointerAddressSpace() != kGenericAS;
              bool FVAS1 = !FV->getType()->isPointerTy() ||
                           FV->getType()->getPointerAddressSpace() != kGenericAS;
              if (TVAS1 && FVAS1) {
                auto *NewSI =
                    SelectInst::Create(SI->getCondition(), TV, FV, "",
                                       SI->getIterator());
                SmallVector<Use *, 8> Uses;
                for (auto &U : SI->uses())
                  Uses.push_back(&U);
                for (auto *U : Uses)
                  U->set(NewSI);
                ToErase.push_back(SI);
                Progress = true;
              }
            }
          }
        }
      }

      for (auto *I : ToErase)
        I->eraseFromParent();
    }

    if (Progress)
      Changed = true;
  }

  // Handle constant expressions with addrspacecast to/from Generic
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

          unsigned NewDstAS = mapAS(DstAS);

          if (SrcAS == NewDstAS) {
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
