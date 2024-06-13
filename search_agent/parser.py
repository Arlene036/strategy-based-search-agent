import re
from typing import Dict, List, Tuple, Optional
from langchain.output_parsers import ListOutputParser
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from pydantic import BaseModel

class AskUserParser(BaseOutputParser):
    def parse(self, text) -> Dict:
        # pattern = r"Answer:\s*(Clear|Unclear)\s*Question:\s*(.*)"
        pattern = r"Clear Score:\s*(.*)\s*Question:\s*(.*)"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            # answer = match.group(1).strip()
            # question = match.group(2).strip()

            # question = None if question == "None" else question
            
            # return {'Answer': answer, 'Question': question}
            clear_score = match.group(1).strip()
            question = match.group(2).strip()

            if int(clear_score) <= 2:
                return {'Answer': 'Unclear', 'Question': question}
            else:
                return {'Answer': 'Clear', 'Question': None}
        else:
            return {'Answer': 'Clear', 'Question': None}


class StrategySuggestionParser(BaseOutputParser):
    def parse(self, text) -> Tuple[str, str]:
        # strategy
        valid_strategies = ["Parallel", "Planning", "Direct"]
        marker = "Strategy:"
        strategy = None

        marker_position = text.find(marker)
        if marker_position != -1:
            # Extract the text following the marker
            start = marker_position + len(marker)
            end = text.find("\n", start)
            strategy = text[start:end].strip()
        else:
            for valid_strategy in valid_strategies:
                if valid_strategy in text:
                    if strategy is not None:
                        return "Multiple strategies found in text: {text}" # raise ValueError
                    strategy = valid_strategy

        if strategy not in valid_strategies:
            strategy = 'Direct'

        # suggestion
        marker = "Suggestions:"
        suggestions = None
        marker_position = text.find(marker)
        if marker_position != -1:
            # Extract the text following the marker
            start = marker_position + len(marker)
            end = text.find("\n", start)
            suggestions = text[start:end].strip()
        else:
            suggestions = None
            
        return strategy, suggestions
    
class StrategyParser(BaseOutputParser):
    def parse(self, text) -> Dict:
        valid_strategies = ["Parallel", "Planning", "Direct"]
        marker = "Strategy:"
        strategy = None

        marker_position = text.find(marker)
        if marker_position != -1:
            # Extract the text following the marker
            start = marker_position + len(marker)
            end = text.find("\n", start)
            strategy = text[start:end].strip()
        else:
            # 如果找不到，那就从整个text里面找这三个，如果只有一个，就返回这个，如果有多个，就raise error
            for valid_strategy in valid_strategies:
                if valid_strategy in text:
                    if strategy is not None:
                        raise ValueError(f"Multiple strategies found in text: {text}")
                    strategy = valid_strategy

        # Check if the extracted strategy is valid
        if strategy not in valid_strategies:
            # raise ValueError(f"Invalid strategy: {strategy}")
            return 'Direct'

        return strategy
    
class GeneratedQuestionsSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "output_parsers", "list"]

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call and extract questions.

        Args:
            text (str): The text output from an LLM call, containing sections and questions.

        Returns:
            List[str]: A list of strings, each representing a found question in the format 'number.question'.
        """
        parts = text.split("Generated Questions:")

        if len(parts) > 1:
            questions = parts[1].strip().split("\n")
            questions = [q for q in questions if q]

        else:
            parts = text.split("生成的问题:")
            if len(parts) > 1:
                questions = parts[1].strip().split("\n")
                questions = [q for q in questions if q]
            else:
                questions = text.strip().split('\n')
                questions = [q for q in questions if q]

        # Extract questions starting with numbered sequence
        seq_questions = []
        for q in questions:
            if q.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                seq_questions.append(q)
            else:
                break  # Stop extracting if a question doesn't start with numbered sequence
        
        if len(seq_questions) > 5:
            return seq_questions[:5]
        else:
            return seq_questions

    @property
    def _type(self) -> str:
        return "generated questions list"

class RephraseParser(BaseOutputParser):
    def parse(self, text) -> str:
        marker = "Rephrased Question:"
        rephrased_question = None

        marker_position = text.find(marker)
        if marker_position != -1:
            parts = text.split('Rephrased Question:')
            if len(parts) > 1:
                rephrased_question = parts[1].strip()
        else:
            raise ValueError(f"Rephrased question not found in text: {text}")

        return rephrased_question
