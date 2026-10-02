import re
from typing import List, Dict, Optional
from agents.agent import Agent
from agents.money import format_money


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"

    INR_PROMPT = (
        "Estimate the current typical selling price of this product in India, in Indian rupees (INR), "
        "as sold by major Indian online stores on a normal day (not a flash sale, not the MRP). "
        "Respond with the number of rupees only, no currency symbol and no explanation."
    )

    def __init__(self, collection=None, client=None, encoder=None, model: Optional[str] = None):
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma Datastore,
        And setting up the vector encoding model
        :param collection: the USD "products" collection (legacy mode); INR mode passes similars directly
        """
        self.log("Initializing Frontier Agent")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.MODEL = model or "gpt-5.1"
        self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        if encoder is None and collection is not None:
            from agents.inr_store import get_encoder

            encoder = get_encoder()
        self.model = encoder
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(
        self, description: str, similars: List[str], prices: List[float]
    ) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string.
        Handles "$1,299.99" as well as "₹1,29,999", "Rs. 4,499" and "INR 999".
        """
        s = re.sub(r"(₹|\$|INR|Rs\.?)", "", s or "", flags=re.IGNORECASE).replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def make_inr_context(self, similars) -> str:
        if not similars:
            return ""
        message = "For context, here are similar products recently seen on Indian stores, with their prices in INR:\n\n"
        for similar in similars:
            mrp = f" (MRP {format_money(similar.mrp)})" if similar.mrp else ""
            message += f"{similar.document}\nPrice: {format_money(similar.price)}{mrp}\n\n"
        return message

    def estimate_inr(self, description: str, similars) -> Optional[float]:
        """
        Estimate the typical Indian selling price in INR, using similar INR items as context.
        Returns None when the model gives no usable number.
        """
        message = f"{self.INR_PROMPT}\n\nProduct:\n{description}\n\n{self.make_inr_context(similars)}"
        self.log(f"Frontier Agent is asking {self.MODEL} for an INR estimate with {len(similars)} similar items")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": message}],
            seed=42,
            reasoning_effort="none",
        )
        result = self.get_price(response.choices[0].message.content)
        if result <= 0:
            return None
        self.log(f"Frontier Agent completed - estimating {format_money(result)}")
        return result

    def price(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        documents, prices = self.find_similars(description)
        self.log(
            f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products"
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, prices),
            seed=42,
            reasoning_effort="none",
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
