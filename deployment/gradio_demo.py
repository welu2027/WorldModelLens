"""Gradio UI for quick demos."""

import gradio as gr
import numpy as np
import torch

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer


class DemoWorldModel:
    def __init__(self):
        self.wm = None
        self.analyzer = None
        self._setup_model()

    def _setup_model(self):
        from world_model_lens.backends.generic_adapter import WorldModelAdapter

        class SimpleAdapter(WorldModelAdapter):
            def __init__(self):
                super().__init__(None)
                self.encoder = torch.nn.Linear(128, 256)
                self.gru = torch.nn.GRUCell(260, 256)
                self.posterior_net = torch.nn.Linear(256, 32)
                self.prior_net = torch.nn.Linear(256, 32)

            def encode(self, obs, context=None):
                obs_enc = self.encoder(obs)
                return obs_enc, obs_enc

            def dynamics(self, state, action=None):
                if action is not None:
                    gru_input = torch.cat([state.hidden, action], dim=-1)
                else:
                    gru_input = state.hidden
                new_hidden = self.gru(gru_input)
                return new_hidden

            def get_components(self):
                return ["encoder", "gru", "posterior", "prior"]

        adapter = SimpleAdapter()
        self.wm = HookedWorldModel(adapter=adapter, config=WorldModelConfig())
        self.analyzer = BeliefAnalyzer(self.wm)

    def analyze(
        self,
        obs_str: str,
        seq_length: int = 20,
    ):
        try:
            if obs_str.strip():
                obs = np.array([float(x) for x in obs_str.strip().split()])
                obs = torch.tensor(obs, dtype=torch.float32)
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                obs = obs[:128]
                obs = torch.nn.functional.pad(obs, (0, max(0, 128 - len(obs))), value=0)
            else:
                obs = torch.randn(seq_length, 128)

            actions = torch.randn(seq_length, 4)

            traj, cache = self.wm.run_with_cache(obs.unsqueeze(0), actions)

            surprise = self.analyzer.surprise_timeline(cache)

            return (
                f"Trajectory Length: {len(traj)}\n"
                f"Cache Components: {', '.join(cache.component_names)}\n"
                f"Mean Surprise: {surprise.mean_surprise:.4f}\n"
                f"Max Surprise: {surprise.max_surprise_value:.4f}\n"
                f"Max Surprise @ t={surprise.max_surprise_timestep}",
                surprise.kl_sequence.tolist()
                if hasattr(surprise.kl_sequence, "tolist")
                else list(surprise.kl_sequence),
            )
        except Exception as e:
            return f"Error: {str(e)}", []

    def compare_trajectories(
        self,
        traj1_str: str,
        traj2_str: str,
    ):
        try:
            return "Comparison feature: Coming soon!", []
        except Exception as e:
            return f"Error: {str(e)}", []


demo = DemoWorldModel()

with gr.Blocks(title="World Model Lens Demo") as app:
    gr.Markdown("# World Model Lens - Interactive Demo")

    with gr.Tab("Analyze Episode"):
        with gr.Row():
            with gr.Column():
                obs_input = gr.Textbox(
                    label="Observation Vector",
                    placeholder="Enter comma-separated values or leave empty for random",
                    lines=3,
                )
                seq_length = gr.Slider(5, 50, value=20, step=1, label="Sequence Length")
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                results_output = gr.Textbox(label="Results", lines=8)
                plot_output = gr.Plot(label="Surprise Timeline")

        analyze_btn.click(
            fn=demo.analyze,
            inputs=[obs_input, seq_length],
            outputs=[results_output, plot_output],
        )

    with gr.Tab("Compare Trajectories"):
        with gr.Row():
            with gr.Column():
                traj1_input = gr.Textbox(label="Trajectory 1", lines=5)
                traj2_input = gr.Textbox(label="Trajectory 2", lines=5)
                compare_btn = gr.Button("Compare", variant="primary")
            with gr.Column():
                comparison_output = gr.Textbox(label="Comparison", lines=8)

        compare_btn.click(
            fn=demo.compare_trajectories,
            inputs=[traj1_input, traj2_input],
            outputs=[comparison_output],
        )

    gr.Markdown("""
    ## About World Model Lens

    World Model Lens provides interpretability tools for any world model architecture.
    - Works with DreamerV3, TD-MPC2, IRIS, and custom models
    - Supports activation caching, patching, and probing
    - No RL assumptions required

    [Documentation](https://github.com/anomalyco/world_model_lens) | 
    [PyPI](https://pypi.org/project/world_model_lens/)
    """)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
