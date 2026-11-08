"""Exercise JIT-warmed small-angle SE(3) backward, including exact identity."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.pose_prediction import rotation_and_jacobian, se3_exp_batch


class EKFGradientTests(unittest.TestCase):
    def test_float32_taylor_and_regular_branches_remain_finite_after_jit_warmup(self):
        angles=torch.tensor([0.,1e-12,1e-8,1e-6,1e-4,9e-4,1.1e-3,.1])
        initial=torch.stack((angles,angles*.3,-angles*.2),-1)
        with torch.autograd.detect_anomaly():
            for _ in range(12):
                value=initial.clone().requires_grad_()
                rotation,jacobian=rotation_and_jacobian(value)
                weights=torch.arange(9,dtype=value.dtype).reshape(1,3,3)
                gradient,=torch.autograd.grad(((rotation+2*jacobian)*weights).sum(),value)
                self.assertTrue(torch.isfinite(gradient).all())
                torch.testing.assert_close(rotation[0],torch.eye(3),rtol=0,atol=0)
                torch.testing.assert_close(jacobian[0],torch.eye(3),rtol=0,atol=0)
                self.assertGreater(gradient[0].abs().sum().item(),0.)

    def test_identity_se3_exp_keeps_translation_and_rotation_gradients(self):
        for _ in range(8):
            value=torch.zeros(3,6,requires_grad=True)
            output=se3_exp_batch(value)
            weights=torch.arange(16,dtype=torch.float32).reshape(1,4,4)
            gradient,=torch.autograd.grad((output*weights).sum(),value)
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient[:,:3],torch.tensor([[3.,7.,11.]]).expand(3,-1))
            self.assertGreater(gradient[:,3:].abs().sum().item(),0.)


if __name__=="__main__":unittest.main()
