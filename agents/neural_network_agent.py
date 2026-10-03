from typing import Optional

from agents.agent import Agent
from agents.deep_neural_network import DeepNeuralNetworkInference

INR_WEIGHTS = "models/nn_inr.pth"
USD_WEIGHTS = "deep_neural_network.pth"


class NeuralNetworkAgent(Agent):
    """
    Two models behind one agent, chosen by pricer mode:
      usd_legacy  the original USD model (deep_neural_network.pth, hardcoded target stats)
      inr         the INR model from scripts/train_nn_inr.py (models/nn_inr.pth + nn_inr_meta.json)
    """

    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, mode: str = "usd_legacy", weights: Optional[str] = None):
        self.log(f"Neural Network Agent is initializing ({mode})")
        self.mode = mode
        if mode == "inr":
            from agents.deep_neural_network import InrPriceModel

            self.neural_network = InrPriceModel(weights or INR_WEIGHTS)
        else:
            self.neural_network = DeepNeuralNetworkInference()
            self.neural_network.setup()
            self.neural_network.load(weights or USD_WEIGHTS)
        self.log("Neural Network Agent is ready and weights are loaded")

    def price(self, description: str, category: Optional[str] = None) -> float:
        """
        Estimate the price of the described item, in dollars (usd_legacy) or rupees (inr)
        :param description: the product to be estimated
        :param category: the app's category, used by the INR model when its title rules find none
        :return: the price as a float
        """
        self.log("Neural Network Agent is starting a prediction")
        if self.mode == "inr":
            result = self.neural_network.price(description, category)
            self.log(f"Neural Network Agent completed - predicting Rs {result:,.0f}")
            return result
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
