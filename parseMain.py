import argparse

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--weight_decay", type=int, default=0.0)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--upload_model",type=str, default="olive-night-782.pth")

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate for the model training")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=int, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.0005,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    parser.add_argument("--device", type=int, default=0,
                    help="Determines which Nvidia device will use (just one number)")

    return parser