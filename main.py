from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from anotherJEPA import JEPA

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    state_dim = 128  # Should match the dimension used during training
    action_dim = 2
    hidden_dim = 128
    ema_rate = 0.99

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JEPA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, ema_rate=ema_rate).to(device)

    # Load the saved weights
    model_path = "best_model.pth"
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path} on {device}")
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Initializing a new model on {device}.")

    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
