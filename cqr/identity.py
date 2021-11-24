from typing import Optional

from .cqr import ConversationalQueryRewriter

__all__ = ["Identity"]


class Identity(ConversationalQueryRewriter):
    """Identity CQR"""

    def __init__(self):
        super().__init__("Identity")

    def rewrite(self, query: str, context: Optional[str] = None) -> str:
        self.turn_id += 1
        return self.pre_process(query)

    def reset_history(self):
        super().reset_history()
        self.history = []
        
    @staticmethod
    def pre_process(text):
        text = re.sub(
            r"[\[\]\(\){},`;:=$#@%|_~^<>+*\s\n\t\\/]+\ *",
            " ",
            text.encode('ascii', 'replace').decode(),
        )
        return text