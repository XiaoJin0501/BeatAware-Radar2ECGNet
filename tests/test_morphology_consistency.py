import unittest

import torch

from src.losses.morphology_consistency import morphology_consistency_loss


class MorphologyConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.gt = torch.linspace(0.2, 0.8, 32).reshape(1, 1, 32)
        self.bounds = torch.tensor([[0, 32]])
        self.shape = ((self.gt - 0.2) / 0.6).clone().requires_grad_()

    def test_consistent_predictions(self):
        loss, count = morphology_consistency_loss(self.gt, self.shape, self.gt, self.bounds)
        self.assertEqual(count, 1)
        self.assertLess(loss.item(), 1e-6)

    def test_offset_not_hidden_and_head_detached(self):
        pred = (self.gt + 0.06).requires_grad_()
        loss, _ = morphology_consistency_loss(pred, self.shape, self.gt, self.bounds)
        self.assertAlmostEqual(loss.item(), 0.1, places=5)
        loss.backward()
        self.assertGreater(pred.grad.abs().sum().item(), 0)
        self.assertIsNone(self.shape.grad)

    def test_reference_control_independent_of_head(self):
        loss, _ = morphology_consistency_loss(self.gt, self.shape * 0, self.gt,
                                              self.bounds, target_mode="reference")
        self.assertLess(loss.item(), 1e-6)

    def test_invalid_and_flat_cycles_have_no_fabricated_target(self):
        for bounds, gt in [(torch.tensor([[-1, 32]]), self.gt),
                           (torch.tensor([[0, 33]]), self.gt),
                           (torch.tensor([[0, 10]]), self.gt),
                           (self.bounds, torch.ones_like(self.gt))]:
            pred = self.gt.clone().requires_grad_()
            loss, count = morphology_consistency_loss(pred, self.shape, gt, bounds)
            self.assertEqual(count, 0)
            loss.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertEqual(pred.grad.abs().sum().item(), 0)

    def test_mismatched_reference_rejected(self):
        with self.assertRaises(ValueError):
            morphology_consistency_loss(self.gt, self.shape, self.gt[..., :-1], self.bounds)

    def test_real_remnet_forward_backward_for_both_channel_counts(self):
        from src.models.remalign.rem_align_net import REMAlignMTLNet

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for channels in (1, 50):
                with self.subTest(channels=channels):
                    model = REMAlignMTLNet(in_channels=channels, use_alignment=False).eval()
                    n_params = sum(p.numel() for p in model.parameters())
                    output, _ = model(torch.randn(1, channels, 71, 120))
                    reference = torch.linspace(0.1, 0.9, 1600).reshape(1, 1, 1600)
                    loss, count = morphology_consistency_loss(
                        output, output.shape200_pred, reference, torch.tensor([[600, 1000]])
                    )
                    loss.backward()
                    self.assertEqual(count, 1)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(any(p.grad is not None for p in model.full_decoder.parameters()))
                    self.assertTrue(all(p.grad is None for p in model.shape_decoder.parameters()))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None))
                    self.assertEqual(n_params, sum(p.numel() for p in model.parameters()))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
