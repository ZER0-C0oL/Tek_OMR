import json

def find_marks(user_ans, correct_ans):
    print(type(correct_ans))
    with open(correct_ans) as f:
        correct_ans = json.loads(f.read())
    print(type(correct_ans))
    marks = 0
    options = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    for x in user_ans:
        if correct_ans[x] == options[list(user_ans[x].values()).index(True)]:
            marks += 1

    return marks