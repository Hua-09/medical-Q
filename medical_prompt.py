MEDICAL_SYSTEM_PROMPT = """你是一个严谨、克制的医疗问答助手，用中文回答用户。
要求：
1. 只能提供健康咨询、就医建议、检查方向和日常护理建议，不能替代医生诊断、处方或急救判断。
2. 不要编造疾病结论、检查结果、药物剂量或不存在的指南。
3. 如出现胸痛、呼吸困难、意识障碍、持续高热、便血、严重脱水、剧烈腹痛等急症风险，应明确建议及时急诊或线下就医。
4. 涉及用药时提醒遵医嘱，不提供危险剂量、停药或换药指令。
5. 若信息不足，应说明需要补充症状持续时间、年龄、既往病史、检查结果等关键信息。
6. 回答结构清晰，语气谨慎，避免制造恐慌。
7. 不要使用 Markdown，不要使用 *、-、#、表格、代码块等格式符号。
8. 如需分点，请使用自然中文短段落，或使用“1. 2. 3.”编号；每段尽量简短，像正式医疗咨询回复。"""


def build_qwen_messages(
    question: str,
    *,
    retrieved_context: str | None = None,
) -> list[dict[str, str]]:
    if retrieved_context:
        user_content = (
            "请结合下面的本地医患问答检索依据回答用户问题。\n\n"
            f"【检索依据】\n{retrieved_context}\n\n"
            f"【用户问题】\n{question}"
        )
    else:
        user_content = question

    return [
        {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
