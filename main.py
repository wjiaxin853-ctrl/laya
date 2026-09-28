import laya


model_path = "/Users/wjx/Desktop/modles/laya"

agent= laya.load(model_path)

# 状态，发来的邮件内容，用户要撤销退款，需要模型决策
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

questions = {
    # 选择：问题到那个部门处理
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    # 打分：紧急程度
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    # 判断：用户流失风险
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    # 判断：用户是否明确要求退款
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

result= agent.predict(state, questions)
print(result)

