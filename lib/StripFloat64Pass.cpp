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

// When -fp64=0, removes functions that use double-precision floating point.
// chipStar/HIP test binaries compile ALL type variants (int, float, double)
// into one SPIR-V module. The double variants trigger Float64 capability
// which causes the entire module to fail on devices without fp64 support.
// This pass removes the dead double code so non-double kernels can load.

#include "StripFloat64Pass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "clspv/Option.h"

using namespace llvm;

// Check if a type contains double anywhere in its structure.
static bool containsDouble(Type *Ty) {
  if (Ty->isDoubleTy())
    return true;
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    for (unsigned i = 0; i < ST->getNumElements(); i++) {
      if (containsDouble(ST->getElementType(i)))
        return true;
    }
  }
  if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    return containsDouble(AT->getElementType());
  }
  if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    return containsDouble(VT->getElementType());
  }
  return false;
}

// Check if a function uses double in its signature or body.
static bool usesDouble(Function &F) {
  // Check return type.
  if (containsDouble(F.getReturnType()))
    return true;

  // Check parameter types.
  for (auto &Arg : F.args()) {
    if (containsDouble(Arg.getType()))
      return true;
  }

  // Check instruction types in the body.
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (containsDouble(I.getType()))
        return true;
      // Check operand types.
      for (auto &Op : I.operands()) {
        if (containsDouble(Op->getType()))
          return true;
      }
    }
  }

  return false;
}

// Replace a function's body with a simple return of zero/void.
static void stubFunction(Function &F) {
  while (!F.empty())
    F.back().eraseFromParent();

  auto *BB = BasicBlock::Create(F.getContext(), "entry", &F);
  IRBuilder<> Builder(BB);

  Type *RetTy = F.getReturnType();
  if (RetTy->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(Constant::getNullValue(RetTy));
  }
}

PreservedAnalyses
clspv::StripFloat64Pass::run(Module &M, ModuleAnalysisManager &) {
  // Only strip when fp64 is disabled.
  if (clspv::Option::FP64())
    return PreservedAnalyses::all();

  bool Changed = false;
  SmallVector<Function *, 16> ToStub;

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (usesDouble(F)) {
      ToStub.push_back(&F);
    }
  }

  for (auto *F : ToStub) {
    stubFunction(*F);
    Changed = true;
  }

  // Remove declarations of double-using functions that are now unused.
  SmallVector<Function *, 8> ToRemove;
  for (auto &F : M) {
    if (!F.isDeclaration())
      continue;
    if (F.use_empty() && usesDouble(F)) {
      ToRemove.push_back(&F);
    }
  }
  for (auto *F : ToRemove) {
    F->eraseFromParent();
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
