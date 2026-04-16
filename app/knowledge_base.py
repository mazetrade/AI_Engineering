import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# We use a small but powerful model — runs locally, no extra API cost
# It converts text into vectors (lists of numbers) that capture meaning
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------------------------ #
#  PRODUCT & POLICY DATA                                               #
#  In a real app this would come from a database or CMS               #
#  For now we define it here so you can see exactly how it works      #
# ------------------------------------------------------------------ #

KNOWLEDGE_DOCUMENTS = [
    # --- Products ---
    {
        "id": "prod_001",
        "category": "product",
        "content": "Nike Air Max 270 is available in sizes 38 to 47. "
                   "Colors: black/white, all-white, navy blue. Price: 149.99 USD. "
                   "In stock. Free delivery over 50 USD."
    },
    {
        "id": "prod_002",
        "category": "product",
        "content": "Samsung 55-inch 4K Smart TV model QN55Q80C. Price: 799.99 USD. "
                   "Features: QLED display, HDR10+, built-in Alexa. "
                   "In stock. Delivery in 3-5 business days."
    },
    {
        "id": "prod_003",
        "category": "product",
        "content": "Apple AirPods Pro (2nd generation). Price: 249.99 USD. "
                   "Features: Active Noise Cancellation, Transparency mode, "
                   "30h battery with case. In stock."
    },
    {
        "id": "prod_004",
        "category": "product",
        "content": "Levi's 501 Original Jeans. Sizes: 28x30 to 40x34. "
                   "Colors: dark wash, light wash, black. Price: 69.99 USD. "
                   "In stock. Free returns within 30 days."
    },
    {
        "id": "prod_005",
        "category": "product",
        "content": "KitchenAid Stand Mixer 5-Quart Artisan Series. "
                   "Colors: red, silver, black, white. Price: 379.99 USD. "
                   "In stock. 1-year manufacturer warranty included."
    },

    # --- Shipping policies ---
    {
        "id": "ship_001",
        "category": "shipping",
        "content": "Standard shipping takes 5-7 business days and costs 5.99 USD. "
                   "Free standard shipping on orders over 50 USD."
    },
    {
        "id": "ship_002",
        "category": "shipping",
        "content": "Express shipping takes 2-3 business days and costs 12.99 USD. "
                   "Overnight shipping costs 24.99 USD and arrives next business day "
                   "if ordered before 2pm EST."
    },
    {
        "id": "ship_003",
        "category": "shipping",
        "content": "We ship to the US, Canada, UK, France, Germany, Spain, "
                   "Italy, Australia, and Japan. International orders may incur "
                   "customs fees charged by the destination country."
    },

    # --- Return policies ---
    {
        "id": "ret_001",
        "category": "returns",
        "content": "Items can be returned within 30 days of delivery. "
                   "Products must be unused and in original packaging. "
                   "Refunds are processed within 5-7 business days after we receive the item."
    },
    {
        "id": "ret_002",
        "category": "returns",
        "content": "Electronics must be returned within 15 days. "
                   "Opened software, digital downloads, and customized items "
                   "are not eligible for returns."
    },
    {
        "id": "ret_003",
        "category": "returns",
        "content": "To start a return, go to My Orders in your account, "
                   "select the item, and click Return Item. "
                   "You will receive a prepaid return label by email within 24 hours."
    },

    # --- Payment policies ---
    {
        "id": "pay_001",
        "category": "payment",
        "content": "We accept Visa, Mastercard, American Express, PayPal, "
                   "Apple Pay, and Google Pay. "
                   "All transactions are secured with 256-bit SSL encryption."
    },
    {
        "id": "pay_002",
        "category": "payment",
        "content": "Buy Now Pay Later is available through Klarna and Afterpay. "
                   "Split your purchase into 4 interest-free payments. "
                   "Available on orders between 35 USD and 1500 USD."
    },
    {
        "id": "pay_003",
        "category": "payment",
        "content": "If your payment fails, please check that your billing address "
                   "matches your card statement, and that your card has not expired. "
                   "Contact your bank if the issue persists."
    },

    # --- Order tracking ---
    {
        "id": "ord_001",
        "category": "orders",
        "content": "Once your order ships, you will receive a tracking email "
                   "with a link to track your package in real time. "
                   "Orders are typically processed within 1 business day."
    },
    {
        "id": "ord_002",
        "category": "orders",
        "content": "You can view all your orders and their statuses by logging into "
                   "your account and visiting the My Orders section. "
                   "Guest orders can be tracked using the order number and email address."
    },
]


class KnowledgeBase:
    """
    Manages the vector store for semantic search over our documents.

    How it works:
    1. Each document is converted to a vector (embedding) using a language model
    2. All vectors are stored in a FAISS index — an ultra-fast similarity search engine
    3. When a user asks a question, we embed their question the same way
    4. FAISS finds the documents whose vectors are closest to the question vector
    5. Those documents are returned as context for Claude
    """

    def __init__(self):
        self.documents = KNOWLEDGE_DOCUMENTS
        self.index = None
        self.embeddings = None
        self._build_index()

    def _build_index(self):
        """
        Converts all documents to vectors and loads them into FAISS.
        This runs once when the app starts.
        """
        print("Building knowledge base index...")

        # Extract just the text content from each document
        texts = [doc["content"] for doc in self.documents]

        # Convert all texts to vectors at once (batch processing = faster)
        # Shape: (num_documents, embedding_dimension) e.g. (16, 384)
        self.embeddings = EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)

        # Get the vector dimension (384 for all-MiniLM-L6-v2)
        dimension = self.embeddings.shape[1]

        # Create a FAISS index using L2 (Euclidean) distance
        # IndexFlatL2 = exact search, no approximation — perfect for small datasets
        self.index = faiss.IndexFlatL2(dimension)

        # Add all document vectors to the index
        # FAISS requires float32 format
        self.index.add(self.embeddings.astype(np.float32))

        print(f"Knowledge base ready: {len(self.documents)} documents indexed.")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Finds the most relevant documents for a given query.

        Args:
            query:  The user's question in plain text
            top_k:  How many documents to return (default: 3)

        Returns:
            A list of the top_k most relevant document dicts
        """
        # Convert the query to a vector using the same model
        query_vector = EMBEDDING_MODEL.encode([query], convert_to_numpy=True)

        # Search the FAISS index
        # Returns: distances (how similar) and indices (which documents)
        distances, indices = self.index.search(
            query_vector.astype(np.float32), top_k
        )

        # Build result list — filter out any invalid indices
        results = []
        for idx in indices[0]:
            if idx != -1:  # FAISS returns -1 if fewer results than top_k exist
                results.append(self.documents[idx])

        return results

    def format_context(self, documents: list[dict]) -> str:
        """
        Formats retrieved documents into a clean text block
        that Claude can read as part of its prompt.
        """
        if not documents:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for doc in documents:
            context_parts.append(
                f"[{doc['category'].upper()}]\n{doc['content']}"
            )

        return "\n\n".join(context_parts)


# Create ONE shared instance — the index is built once at startup
knowledge_base = KnowledgeBase()