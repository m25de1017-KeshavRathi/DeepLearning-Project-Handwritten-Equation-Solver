import re

def solve_equation(equation: str) -> float:
    """
    Solves a simple arithmetic equation string.
    Supports: +, -, *, /
    Respects order of operations.
    """
    if not equation:
        return 0.0

    # Add spaces around operators to make splitting easier
    equation = re.sub(r'([*\\/+\\-])', r' \\1 ', equation)
    tokens = equation.split()

    # Shunting-yard algorithm
    values = []
    ops = []

    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    def apply_op():
        op = ops.pop()
        right = values.pop()
        left = values.pop()
        if op == '+':
            values.append(left + right)
        elif op == '-':
            values.append(left - right)
        elif op == '*':
            values.append(left * right)
        elif op == '/':
            if right == 0:
                raise ValueError("Division by zero")
            values.append(left / right)

    for token in tokens:
        if token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
            values.append(float(token))
        elif token in precedence:
            while ops and ops[-1] in precedence and precedence[ops[-1]] >= precedence[token]:
                apply_op()
            ops.append(token)
        elif token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                apply_op()
            if ops and ops[-1] == '(':
                ops.pop()
            else:
                raise ValueError("Mismatched parentheses")

    while ops:
        apply_op()

    if len(values) == 1 and not ops:
        return values[0]
    else:
        raise ValueError("Invalid expression")

if __name__ == '__main__':
    # Example usage:
    print(f"Solving '10+2*6' = {solve_equation('10+2*6')}")
    print(f"Solving '100 * 2 + 12' = {solve_equation('100 * 2 + 12')}")
    print(f"Solving '100 * ( 2 + 12 )' = {solve_equation('100 * ( 2 + 12 )')}")
    print(f"Solving '100 * ( 2 + 12 ) / 14' = {solve_equation('100 * ( 2 + 12 ) / 14')}")
