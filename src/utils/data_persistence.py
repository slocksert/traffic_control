import json
import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path


class DataPersistence:
    """
    Handles saving and loading of models, metrics, and training data
    """

    def __init__(self, base_path: str = "data"):
        """
        Initialize data persistence manager

        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "trained_models"
        self.metrics_path = self.base_path / "metrics"
        self.logs_path = self.base_path / "logs"

        for path in [
            self.base_path,
            self.models_path,
            self.metrics_path,
            self.logs_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_q_learning_model(
        self, agent, filename: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Save Q-Learning agent model

        Args:
            agent: Q-Learning agent instance
            filename: Filename for the model (without extension)
            metadata: Additional metadata to save

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename.endswith(".json"):
            filename = f"{filename}_{timestamp}.json"

        filepath = self.models_path / filename

        model_data = {
            "type": "q_learning",
            "timestamp": timestamp,
            "q_table": {key: values.tolist() for key, values in agent.q_table.items()},
            "parameters": {
                "state_space_size": agent.state_space_size,
                "action_space_size": agent.action_space_size,
                "learning_rate": agent.learning_rate,
                "discount_factor": agent.discount_factor,
                "epsilon": agent.epsilon,
                "epsilon_min": agent.epsilon_min,
                "epsilon_decay": agent.epsilon_decay,
            },
            "statistics": {
                "episode_rewards": agent.episode_rewards,
                "episode_lengths": agent.episode_lengths,
                "training_step": agent.training_step,
                "total_steps": agent.total_steps,
            },
            "metadata": metadata or {},
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"Q-Learning model saved to {filepath}")
        return str(filepath)

    def load_q_learning_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load Q-Learning agent model

        Args:
            filepath: Path to model file

        Returns:
            Model data dictionary
        """
        filepath = Path(filepath)
        if not filepath.exists():
            filepath = self.models_path / filepath.name
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "r") as f:
            model_data = json.load(f)

        print(f"Q-Learning model loaded from {filepath}")
        return model_data

    def save_metrics(self, metrics_data: Dict[str, Any], filename: str) -> str:
        """
        Save training metrics

        Args:
            metrics_data: Metrics dictionary
            filename: Filename for metrics

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename.endswith(".json"):
            filename = f"{filename}_{timestamp}.json"

        filepath = self.metrics_path / filename

        metrics_data["timestamp"] = timestamp

        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2, default=self._json_serializer)

        print(f"Metrics saved to {filepath}")
        return str(filepath)

    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Load training metrics

        Args:
            filepath: Path to metrics file

        Returns:
            Metrics dictionary
        """
        filepath = Path(filepath)
        if not filepath.exists():
            filepath = self.metrics_path / filepath.name
            if not filepath.exists():
                raise FileNotFoundError(f"Metrics file not found: {filepath}")

        with open(filepath, "r") as f:
            metrics_data = json.load(f)

        print(f"Metrics loaded from {filepath}")
        return metrics_data

    def save_experiment_log(
        self, experiment_data: Dict[str, Any], experiment_name: str
    ) -> str:
        """
        Save complete experiment log including all runs

        Args:
            experiment_data: Complete experiment data
            experiment_name: Name of the experiment

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.logs_path / filename

        experiment_data["experiment_name"] = experiment_name
        experiment_data["timestamp"] = timestamp
        experiment_data["saved_at"] = datetime.datetime.now().isoformat()

        with open(filepath, "w") as f:
            json.dump(experiment_data, f, indent=2, default=self._json_serializer)

        print(f"Experiment log saved to {filepath}")
        return str(filepath)

    def load_experiment_log(self, filepath: str) -> Dict[str, Any]:
        """
        Load experiment log

        Args:
            filepath: Path to experiment log file

        Returns:
            Experiment data dictionary
        """
        filepath = Path(filepath)
        if not filepath.exists():
            filepath = self.logs_path / filepath.name
            if not filepath.exists():
                raise FileNotFoundError(f"Experiment log not found: {filepath}")

        with open(filepath, "r") as f:
            experiment_data = json.load(f)

        print(f"Experiment log loaded from {filepath}")
        return experiment_data

    def save_comparison_results(
        self, comparison_data: Dict[str, Any], filename: str
    ) -> str:
        """
        Save agent comparison results

        Args:
            comparison_data: Comparison results
            filename: Filename for results

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename.endswith(".json"):
            filename = f"{filename}_{timestamp}.json"

        filepath = self.logs_path / filename

        comparison_data["timestamp"] = timestamp
        comparison_data["saved_at"] = datetime.datetime.now().isoformat()

        with open(filepath, "w") as f:
            json.dump(comparison_data, f, indent=2, default=self._json_serializer)

        print(f"Comparison results saved to {filepath}")
        return str(filepath)

    def list_saved_models(self) -> List[Dict[str, str]]:
        """
        List all saved models

        Returns:
            List of model information dictionaries
        """
        models = []
        for filepath in self.models_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                models.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "type": data.get("type", "unknown"),
                        "timestamp": data.get("timestamp", "unknown"),
                        "size": f"{filepath.stat().st_size / 1024:.1f} KB",
                    }
                )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return sorted(models, key=lambda x: x["timestamp"], reverse=True)

    def list_metrics_files(self) -> List[Dict[str, str]]:
        """
        List all saved metrics files

        Returns:
            List of metrics file information
        """
        metrics = []
        for filepath in self.metrics_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                metrics.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "timestamp": data.get("timestamp", "unknown"),
                        "size": f"{filepath.stat().st_size / 1024:.1f} KB",
                    }
                )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return sorted(metrics, key=lambda x: x["timestamp"], reverse=True)

    def list_experiment_logs(self) -> List[Dict[str, str]]:
        """
        List all experiment logs

        Returns:
            List of experiment log information
        """
        logs = []
        for filepath in self.logs_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                logs.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "experiment_name": data.get("experiment_name", "unknown"),
                        "timestamp": data.get("timestamp", "unknown"),
                        "size": f"{filepath.stat().st_size / 1024:.1f} KB",
                    }
                )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return sorted(logs, key=lambda x: x["timestamp"], reverse=True)

    def backup_model(self, model_path: str, backup_name: Optional[str] = None) -> str:
        """
        Create backup of a model

        Args:
            model_path: Path to model to backup
            backup_name: Optional backup name

        Returns:
            Path to backup file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if backup_name is None:
            backup_name = f"backup_{model_path.stem}"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_name}_{timestamp}.json"
        backup_path = self.models_path / "backups" / backup_filename

        backup_path.parent.mkdir(exist_ok=True)

        import shutil

        shutil.copy2(model_path, backup_path)

        print(f"Model backed up to {backup_path}")
        return str(backup_path)

    def clean_old_files(self, days_old: int = 30) -> None:
        """
        Clean up old files

        Args:
            days_old: Remove files older than this many days
        """
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days_old)

        for directory in [self.models_path, self.metrics_path, self.logs_path]:
            for filepath in directory.glob("*.json"):
                if filepath.stat().st_mtime < cutoff_time.timestamp():
                    print(f"Removing old file: {filepath}")
                    filepath.unlink()

    def export_training_data(self, output_format: str = "csv") -> str:
        """
        Export training data in specified format

        Args:
            output_format: Export format ('csv', 'json')

        Returns:
            Path to exported file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format == "csv":
            import pandas as pd

            all_metrics = []
            for filepath in self.metrics_path.glob("*.json"):
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    if "episode_history" in data:
                        df = pd.DataFrame(data["episode_history"])
                        df["source_file"] = filepath.name
                        all_metrics.append(df)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

            if all_metrics:
                combined_df = pd.concat(all_metrics, ignore_index=True)
                output_path = self.base_path / f"training_data_export_{timestamp}.csv"
                combined_df.to_csv(output_path, index=False)
                print(f"Training data exported to {output_path}")
                return str(output_path)

        elif output_format == "json":
            export_data = {
                "export_timestamp": timestamp,
                "models": self.list_saved_models(),
                "metrics": self.list_metrics_files(),
                "experiments": self.list_experiment_logs(),
            }

            output_path = self.base_path / f"full_export_{timestamp}.json"
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=self._json_serializer)

            print(f"Full data export saved to {output_path}")
            return str(output_path)

        else:
            raise ValueError(f"Unsupported export format: {output_format}")

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and other objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about storage usage

        Returns:
            Storage information dictionary
        """

        def get_dir_size(path):
            total = 0
            for filepath in path.rglob("*"):
                if filepath.is_file():
                    total += filepath.stat().st_size
            return total

        info = {
            "base_path": str(self.base_path),
            "directories": {
                "models": {
                    "path": str(self.models_path),
                    "size_mb": get_dir_size(self.models_path) / (1024 * 1024),
                    "file_count": len(list(self.models_path.glob("*.json"))),
                },
                "metrics": {
                    "path": str(self.metrics_path),
                    "size_mb": get_dir_size(self.metrics_path) / (1024 * 1024),
                    "file_count": len(list(self.metrics_path.glob("*.json"))),
                },
                "logs": {
                    "path": str(self.logs_path),
                    "size_mb": get_dir_size(self.logs_path) / (1024 * 1024),
                    "file_count": len(list(self.logs_path.glob("*.json"))),
                },
            },
            "total_size_mb": get_dir_size(self.base_path) / (1024 * 1024),
        }

        return info
