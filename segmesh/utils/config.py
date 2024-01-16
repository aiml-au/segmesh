import yaml
from typing import Any, Dict

class ConfigLoader:
    """
    A utility class to load configuration parameters from a YAML file.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ConfigLoader with the path to the configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration parameters from the YAML file.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration parameters.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            print(f"Error: Config file {self.config_path} not found.")
            return {}
        except yaml.YAMLError as exc:
            print(f"Error parsing the config file {self.config_path}. Details: {exc}")
            return {}

# Usage
# config_loader = ConfigLoader("config/s3dis_config.yaml")
# config = config_loader.load_config()
