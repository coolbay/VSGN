// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#include "SigmoidFocalLoss.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
}
