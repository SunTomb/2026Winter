"""
Phase 0: SFT Warmup for Memory Manager.
Teaches the Memory Manager the correct output format (ADD/UPDATE/DELETE/NOOP).

Expanded from ~80 duplicated examples to 200+ unique, diverse examples covering:
- Multi-turn dialogue with rich memory context
- Conflict resolution and fact updates
- Temporal changes (location, job, preference shifts)
- Entity disambiguation
- NOOP for irrelevant/meta inputs
- DELETE for resolved goals / outdated info
- Chinese and English mixed scenarios

Usage:
    python scripts/train_phase0_sft.py --model_name Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
import json
import random

import torch
from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer


# ============================================================
# SFT Data: 200+ diverse examples
# ============================================================

_SFT_RAW_EXAMPLES = [
    # ========== ADD — Basic facts ==========
    {
        "memory": "(empty memory)",
        "event": "User says: I'm allergic to peanuts and shellfish.",
        "operation": "ADD: User has food allergies - peanuts and shellfish."
    },
    {
        "memory": "(empty memory)",
        "event": "User says: My name is Alice and I'm a software engineer.",
        "operation": "ADD: User's name is Alice. User works as a software engineer."
    },
    {
        "memory": "(empty memory)",
        "event": "User says: I live in Shanghai with my wife and two kids.",
        "operation": "ADD: User lives in Shanghai with wife and two children."
    },
    {
        "memory": "[m1] User prefers morning meetings",
        "event": "User mentions they started a new job in data science.",
        "operation": "ADD: User works in data science (new job)."
    },
    {
        "memory": "[m1] User likes Python",
        "event": "User says: I've been learning Rust lately and really enjoy it.",
        "operation": "ADD: User is learning Rust and enjoys it."
    },
    {
        "memory": "[m1] User is a student",
        "event": "User says: I just got accepted into Stanford for my PhD in NLP!",
        "operation": "ADD: User accepted into Stanford PhD program in NLP."
    },
    {
        "memory": "(empty memory)",
        "event": "User says: I have a golden retriever named Max.",
        "operation": "ADD: User has a golden retriever named Max."
    },
    {
        "memory": "[m1] User likes Italian food",
        "event": "User says: I've also discovered I love Thai cuisine.",
        "operation": "ADD: User also enjoys Thai cuisine."
    },
    {
        "memory": "[m1] User lives in Beijing\n[m2] User works at Company A",
        "event": "User says: My daughter just turned 5 yesterday!",
        "operation": "ADD: User has a daughter who just turned 5."
    },
    {
        "memory": "(empty memory)",
        "event": "User says: I'm vegetarian for ethical reasons.",
        "operation": "ADD: User is vegetarian for ethical reasons."
    },
    # ========== ADD — Preferences ==========
    {
        "memory": "(empty memory)",
        "event": "User says: I prefer dark mode in all my applications.",
        "operation": "ADD: User prefers dark mode in applications."
    },
    {
        "memory": "[m1] User uses macOS",
        "event": "User says: I always use VSCode with Vim keybindings.",
        "operation": "ADD: User uses VSCode with Vim keybindings."
    },
    {
        "memory": "[m1] User likes jazz\n[m2] User plays guitar",
        "event": "User says: I've been getting into electronic music production lately.",
        "operation": "ADD: User is getting into electronic music production."
    },
    {
        "memory": "[m1] User prefers email communication",
        "event": "User says: For urgent matters, please use Slack instead of email.",
        "operation": "ADD: User prefers Slack for urgent matters."
    },
    {
        "memory": "(empty memory)",
        "event": "User says: I always wake up at 5am and go for a morning run.",
        "operation": "ADD: User wakes up at 5am daily and runs in the morning."
    },
    # ========== ADD — Procedures / Rules ==========
    {
        "memory": "[m1] User manages a team of 8 engineers",
        "event": "User says: Our team does code reviews through GitHub PRs before merging.",
        "operation": "ADD: User's team does code reviews via GitHub PRs before merging."
    },
    {
        "memory": "[m1] User works on ML projects",
        "event": "User says: We always use Weights & Biases for experiment tracking.",
        "operation": "ADD: User's team uses Weights & Biases for experiment tracking."
    },
    {
        "memory": "[m1] User teaches a university course",
        "event": "User says: I give quizzes every other Wednesday in my class.",
        "operation": "ADD: User gives quizzes every other Wednesday in their class."
    },
    # ========== ADD — Relationships ==========
    {
        "memory": "[m1] User lives in New York",
        "event": "User says: My sister Sarah just got married to Tom.",
        "operation": "ADD: User's sister Sarah married Tom."
    },
    {
        "memory": "[m1] User has a dog named Max\n[m2] User lives alone",
        "event": "User says: My best friend Mike is visiting next week.",
        "operation": "ADD: User's best friend Mike is visiting next week."
    },
    {
        "memory": "[m1] User works at Google",
        "event": "User says: My manager Lisa is very supportive of my research work.",
        "operation": "ADD: User's manager Lisa is supportive of their research."
    },
    # ========== UPDATE — Fact corrections ==========
    {
        "memory": "[m1] User lives in Beijing\n[m2] User works at Company A",
        "event": "User says: I just moved to Shanghai last week.",
        "operation": "UPDATE m1: User lives in Shanghai (moved from Beijing recently)."
    },
    {
        "memory": "[m1] User prefers tea\n[m2] User's favorite color is blue",
        "event": "User says: I've switched to coffee recently, tea doesn't wake me up anymore.",
        "operation": "UPDATE m1: User now prefers coffee over tea."
    },
    {
        "memory": "[m1] User works at Company A as a junior developer\n[m2] User likes Python",
        "event": "User says: I got promoted to Senior Engineer last Friday!",
        "operation": "UPDATE m1: User works at Company A as a Senior Engineer (recently promoted)."
    },
    {
        "memory": "[m1] User is single\n[m2] User lives in an apartment",
        "event": "User says: I just got engaged to my partner Emma!",
        "operation": "UPDATE m1: User is engaged to Emma."
    },
    {
        "memory": "[m1] User drives a Honda Civic\n[m2] User commutes 30 minutes",
        "event": "User says: I sold my car and now take the subway to work.",
        "operation": "UPDATE m1: User no longer has a car; takes subway to work."
    },
    {
        "memory": "[m1] User's phone number is 123-456-7890",
        "event": "User says: My new phone number is 098-765-4321.",
        "operation": "UPDATE m1: User's phone number is 098-765-4321."
    },
    {
        "memory": "[m1] User is studying machine learning\n[m2] User uses TensorFlow",
        "event": "User says: I've completely switched to PyTorch now.",
        "operation": "UPDATE m2: User now uses PyTorch (switched from TensorFlow)."
    },
    {
        "memory": "[m1] User's salary is $80k\n[m2] User works as a data analyst",
        "event": "User says: I just accepted a new offer for $120k at a different company.",
        "operation": "UPDATE m1: User's salary is now $120k at a new company."
    },
    # ========== UPDATE — Preference changes ==========
    {
        "memory": "[m1] User likes reading sci-fi novels\n[m2] User reads before bed",
        "event": "User says: Lately I've been more into non-fiction, especially biographies.",
        "operation": "UPDATE m1: User now prefers non-fiction / biographies over sci-fi."
    },
    {
        "memory": "[m1] User exercises in the evening\n[m2] User does yoga",
        "event": "User says: I've switched to morning workouts. Much better for my energy.",
        "operation": "UPDATE m1: User now exercises in the morning."
    },
    {
        "memory": "[m1] User prefers Windows\n[m2] User uses Chrome",
        "event": "User says: I switched to macOS last month and I'm loving it.",
        "operation": "UPDATE m1: User now prefers macOS (switched from Windows)."
    },
    # ========== UPDATE — Temporal changes ==========
    {
        "memory": "[m1] User's meeting is on Tuesdays at 2pm\n[m2] User manages 5 people",
        "event": "User says: Starting next week, our team meeting moves to Wednesdays at 3pm.",
        "operation": "UPDATE m1: User's team meeting is on Wednesdays at 3pm (changed from Tuesdays 2pm)."
    },
    {
        "memory": "[m1] User's daughter is 3 years old\n[m2] User lives in Boston",
        "event": "User says: My daughter just started kindergarten! She turned 5 last month.",
        "operation": "UPDATE m1: User's daughter is 5 years old and started kindergarten."
    },
    {
        "memory": "[m1] User is pregnant (3rd trimester)\n[m2] User works part-time",
        "event": "User says: I had the baby! A healthy boy, we named him Oliver.",
        "operation": "UPDATE m1: User gave birth to a son named Oliver."
    },
    # ========== DELETE — Resolved goals ==========
    {
        "memory": "[m1] User is looking for a new apartment\n[m2] User has a cat named Mimi",
        "event": "User says: Great news, I finally signed the lease on my new place!",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User needs to submit tax forms by April 15\n[m2] User's accountant is John",
        "event": "User says: Just finished submitting all my taxes today.",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User is interviewing at 3 companies\n[m2] User wants to switch to ML",
        "event": "User says: I accepted the offer from OpenAI! Starting next month.",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User's car needs an oil change\n[m2] User drives a Tesla",
        "event": "User says: Got the car serviced yesterday, all good now.",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User is waiting for visa approval\n[m2] User plans to move to Canada",
        "event": "User says: My visa got approved yesterday!",
        "operation": "DELETE m1"
    },
    # ========== DELETE — Outdated information ==========
    {
        "memory": "[m1] User has a coupon for 20% off at Target (expires March 31)\n[m2] User likes shopping at Target",
        "event": "User says: It's already April, I forgot to use that coupon.",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User's friend Bob is visiting this weekend\n[m2] User lives in Chicago",
        "event": "User says: Bob left yesterday, it was a great visit!",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User is reading 'Dune' by Frank Herbert\n[m2] User reads before bed",
        "event": "User says: Just finished Dune last night. What an ending!",
        "operation": "DELETE m1"
    },
    # ========== NOOP — Irrelevant inputs ==========
    {
        "memory": "[m1] User enjoys hiking\n[m2] User works in AI research",
        "event": "User says: What's the weather like today?",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User prefers dark mode\n[m2] User uses VSCode",
        "event": "User says: Can you help me with this code snippet?",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User lives in NYC\n[m2] User has 2 cats",
        "event": "User says: Tell me a joke.",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User is a teacher\n[m2] User likes cooking",
        "event": "User says: What is the capital of France?",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User prefers Python\n[m2] User builds web apps",
        "event": "User says: How do I sort a list in Python?",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User has a meeting at 3pm\n[m2] User works remotely",
        "event": "User says: Thanks for the info!",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User likes Italian food\n[m2] User's favorite restaurant is Luigi's",
        "event": "User says: Can you summarize this article for me?",
        "operation": "NOOP"
    },
    {
        "memory": "(empty memory)",
        "event": "User says: Hello! Nice to meet you.",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User studies economics\n[m2] User is in their 3rd year",
        "event": "User says: What time is it?",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User's birthday is June 15\n[m2] User is 28 years old",
        "event": "User says: Can you translate this to Spanish?",
        "operation": "NOOP"
    },
    # ========== ADD — Multi-turn / rich context ==========
    {
        "memory": "[m1] User is a PhD student\n[m2] User studies NLP\n[m3] User's advisor is Prof. Zhang",
        "event": "User says: I just submitted my first paper to ACL!",
        "operation": "ADD: User submitted first paper to ACL."
    },
    {
        "memory": "[m1] User lives in Shanghai\n[m2] User works at Alibaba\n[m3] User commutes by subway",
        "event": "User says: I started a side project building an AI chatbot for elderly care.",
        "operation": "ADD: User has a side project - AI chatbot for elderly care."
    },
    {
        "memory": "[m1] User likes hiking\n[m2] User bikes to work\n[m3] User does yoga on weekends\n[m4] User is training for a marathon",
        "event": "User says: I just signed up for a triathlon in September!",
        "operation": "ADD: User signed up for triathlon in September."
    },
    {
        "memory": "[m1] User's daughter is 5\n[m2] User's son is 8\n[m3] User lives in suburban Boston",
        "event": "User says: We just adopted a rescue kitten. The kids named her Luna.",
        "operation": "ADD: User's family adopted a rescue kitten named Luna."
    },
    {
        "memory": "[m1] User is a product manager\n[m2] User works at a fintech startup\n[m3] User manages a team of 6",
        "event": "User says: We just closed our Series B funding round - $20M!",
        "operation": "ADD: User's company closed Series B funding - $20M."
    },
    # ========== UPDATE — Conflicting with multiple memories ==========
    {
        "memory": "[m1] User works at Google\n[m2] User's office is in Mountain View\n[m3] User commutes from San Jose",
        "event": "User says: I left Google and joined a startup in San Francisco. Working remotely now.",
        "operation": "UPDATE m1: User left Google, now works at a startup in San Francisco (remote)."
    },
    {
        "memory": "[m1] User is vegetarian\n[m2] User likes cooking Indian food\n[m3] User avoids gluten",
        "event": "User says: My doctor recommended I start eating fish for omega-3s, so I'm now pescatarian.",
        "operation": "UPDATE m1: User is now pescatarian (was vegetarian, started eating fish per doctor's advice)."
    },
    {
        "memory": "[m1] User studies at MIT\n[m2] User majors in CS\n[m3] User is a junior",
        "event": "User says: I graduated from MIT last month and started my first job!",
        "operation": "UPDATE m3: User graduated from MIT and started first job."
    },
    # ========== ADD — Chinese language examples ==========
    {
        "memory": "(empty memory)",
        "event": "User says: 我来自合肥，目前在中科大读研究生。",
        "operation": "ADD: 用户来自合肥，在中国科学技术大学读研究生。"
    },
    {
        "memory": "[m1] 用户在中科大读书",
        "event": "User says: 我的导师是王教授，研究方向是大语言模型。",
        "operation": "ADD: 用户的导师是王教授，研究方向是大语言模型。"
    },
    {
        "memory": "[m1] 用户喜欢吃火锅\n[m2] 用户是四川人",
        "event": "User says: 我最近在学做日式拉面，感觉很有意思。",
        "operation": "ADD: 用户最近在学做日式拉面。"
    },
    {
        "memory": "[m1] 用户住在北京\n[m2] 用户在字节跳动工作",
        "event": "User says: 我上个月搬到深圳了，换了一份在腾讯的工作。",
        "operation": "UPDATE m1: 用户搬到深圳。"
    },
    {
        "memory": "[m1] 用户正在找房子\n[m2] 用户的预算是5000元/月",
        "event": "User says: 终于找到合适的房子了！已经签了合同。",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] 用户喜欢打篮球\n[m2] 用户是程序员",
        "event": "User says: 今天天气怎么样？",
        "operation": "NOOP"
    },
    # ========== ADD — Complex scenarios ==========
    {
        "memory": "[m1] User prefers concise responses\n[m2] User is a busy executive",
        "event": "User says: When we discuss technical topics, I want detailed explanations with code examples though.",
        "operation": "ADD: User wants detailed technical explanations with code examples (exception to concise preference)."
    },
    {
        "memory": "[m1] User has diabetes\n[m2] User monitors blood sugar daily",
        "event": "User says: My doctor put me on a new medication - Metformin, 500mg twice daily.",
        "operation": "ADD: User takes Metformin 500mg twice daily for diabetes."
    },
    {
        "memory": "[m1] User runs a small bakery\n[m2] User works 6 days a week",
        "event": "User says: I hired two new bakers, so I can finally take weekends off!",
        "operation": "ADD: User hired two new bakers and plans to take weekends off."
    },
    {
        "memory": "[m1] User is a pilot\n[m2] User flies domestic routes",
        "event": "User says: I just got certified on the Boeing 787, so I'll start flying international soon.",
        "operation": "ADD: User certified on Boeing 787, will start international flights."
    },
    # ========== Complex UPDATE edge cases ==========
    {
        "memory": "[m1] User has 3 employees\n[m2] User runs a consulting firm",
        "event": "User says: Two people quit last week, but I just hired four new consultants.",
        "operation": "UPDATE m1: User now has 5 employees (2 left, 4 new hires)."
    },
    {
        "memory": "[m1] User prefers email at work@company.com\n[m2] User works at Company A",
        "event": "User says: I changed companies, my new email is user@newcompany.com.",
        "operation": "UPDATE m1: User's email is user@newcompany.com (changed companies)."
    },
    {
        "memory": "[m1] User is learning French (beginner)\n[m2] User plans to visit Paris",
        "event": "User says: I passed the B2 French exam! I'm now upper intermediate.",
        "operation": "UPDATE m1: User's French level is B2 (upper intermediate)."
    },
    # ========== Complex DELETE ==========
    {
        "memory": "[m1] User has a toothache\n[m2] User has a dentist appointment this Friday\n[m3] User is nervous about the dentist",
        "event": "User says: The dentist fixed my tooth, no more pain! It wasn't as bad as I thought.",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User's laptop is broken\n[m2] User uses a MacBook Pro\n[m3] User needs laptop for work",
        "event": "User says: Got my laptop repaired today. Apple replaced the screen for free under warranty.",
        "operation": "DELETE m1"
    },
    # ========== Tricky NOOP cases ==========
    {
        "memory": "[m1] User lives in Tokyo\n[m2] User speaks Japanese",
        "event": "User says: I love living in Tokyo, the food is amazing.",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User has a dog named Max\n[m2] User likes hiking",
        "event": "User says: Max and I went hiking today, it was beautiful.",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User is a Python developer\n[m2] User works on backend systems",
        "event": "User says: Python is such a versatile language, I love working with it.",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User's birthday is March 15\n[m2] User likes chocolate cake",
        "event": "User says: Happy Wednesday to me! Just another regular day.",
        "operation": "NOOP"
    },
    # ========== More ADD diversity ==========
    {
        "memory": "(empty memory)",
        "event": "User says: I have a latex allergy, please never suggest anything with latex.",
        "operation": "ADD: User has a latex allergy - avoid latex suggestions."
    },
    {
        "memory": "[m1] User is a lawyer",
        "event": "User says: I specialize in intellectual property law, particularly patent disputes.",
        "operation": "ADD: User specializes in intellectual property law, particularly patents."
    },
    {
        "memory": "[m1] User has 2 cats\n[m2] User lives in an apartment",
        "event": "User says: My landlord just told me he's selling the building. I need to move by June.",
        "operation": "ADD: User needs to move by June - landlord selling building."
    },
    {
        "memory": "[m1] User is planning a wedding\n[m2] User's fiancée is Sarah",
        "event": "User says: We decided on a destination wedding in Bali next October.",
        "operation": "ADD: User planning destination wedding in Bali next October."
    },
    {
        "memory": "[m1] User works in cybersecurity\n[m2] User has CISSP certification",
        "event": "User says: I just passed the OSCP certification exam!",
        "operation": "ADD: User passed OSCP certification exam."
    },
    # ========== Entity disambiguation ==========
    {
        "memory": "[m1] User's mother lives in London\n[m2] User's sister lives in Paris",
        "event": "User says: My mother is moving to be closer to my sister.",
        "operation": "UPDATE m1: User's mother is moving to Paris (to be closer to user's sister)."
    },
    {
        "memory": "[m1] User's son Jack plays soccer\n[m2] User's son Tom plays piano",
        "event": "User says: Tom just won first place in a piano competition!",
        "operation": "ADD: User's son Tom won first place in a piano competition."
    },
    {
        "memory": "[m1] User's colleague Alice likes coffee\n[m2] User's friend Alice from college likes tea",
        "event": "User says: My colleague Alice switched to matcha lattes.",
        "operation": "UPDATE m1: User's colleague Alice now drinks matcha lattes (switched from coffee)."
    },
    # ========== Multi-hop reasoning ==========
    {
        "memory": "[m1] User works at Microsoft\n[m2] User is in the Azure team\n[m3] User's project launches in Q2",
        "event": "User says: The CEO just announced our whole division is being restructured. Azure is merging with the AI team.",
        "operation": "ADD: User's Azure division is merging with AI team due to company restructuring."
    },
    {
        "memory": "[m1] User lives in Boston\n[m2] User's child goes to Lincoln Elementary\n[m3] User drives child to school",
        "event": "User says: We're moving to Cambridge next month. Luckily the new house is near a great school.",
        "operation": "UPDATE m1: User is moving to Cambridge next month."
    },
    # ========== More diverse scenarios ==========
    {
        "memory": "(empty memory)",
        "event": "User says: I'm visually impaired, so please use plain text descriptions instead of complex formatting.",
        "operation": "ADD: User is visually impaired - use plain text, avoid complex formatting."
    },
    {
        "memory": "(empty memory)",
        "event": "User says: I work night shifts from 10pm to 6am, so I sleep during the day.",
        "operation": "ADD: User works night shifts (10pm-6am), sleeps during the day."
    },
    {
        "memory": "[m1] User is a freelance designer\n[m2] User uses Figma",
        "event": "User says: I just signed a full-time contract with a major agency. No more freelancing!",
        "operation": "UPDATE m1: User signed full-time contract with an agency (no longer freelancing)."
    },
    {
        "memory": "[m1] User is training for a 5K run\n[m2] User started running 2 months ago",
        "event": "User says: Completed my first 5K today in 28 minutes!",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] User is trying to quit smoking\n[m2] User smokes 10 cigarettes a day",
        "event": "User says: It's been 6 months since my last cigarette! I'm finally smoke-free.",
        "operation": "UPDATE m2: User is now smoke-free (quit 6 months ago)."
    },
    {
        "memory": "[m1] User has a fear of flying",
        "event": "User says: I took my first flight in 10 years last week! I'm getting over my fear.",
        "operation": "UPDATE m1: User is overcoming fear of flying (took first flight in 10 years)."
    },
    # ========== More Chinese examples ==========
    {
        "memory": "(empty memory)",
        "event": "User says: 我对花生过敏，这很严重，请记住。",
        "operation": "ADD: 用户对花生严重过敏。"
    },
    {
        "memory": "[m1] 用户是大二学生\n[m2] 用户学计算机科学",
        "event": "User says: 我决定转到人工智能专业了。",
        "operation": "UPDATE m2: 用户转到人工智能专业（原计算机科学）。"
    },
    {
        "memory": "[m1] 用户在备考GRE\n[m2] 用户计划出国留学",
        "event": "User says: GRE考了330分！终于考完了。",
        "operation": "DELETE m1"
    },
    {
        "memory": "[m1] 用户养了一只猫叫咪咪\n[m2] 用户住在北京",
        "event": "User says: 帮我写一份工作邮件。",
        "operation": "NOOP"
    },
    # ========== Edge Cases ==========
    {
        "memory": "(empty memory)",
        "event": "User says: ",
        "operation": "NOOP"
    },
    {
        "memory": "(empty memory)",
        "event": "User says: ... (silence)",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User prefers Python over Java",
        "event": "User says: I still love Python, it's the best language for data science.",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User lives in Tokyo\n[m2] User speaks Japanese and English",
        "event": "User says: Can you write this in Japanese? 日本語でお願いします。",
        "operation": "NOOP"
    },
    # ========== More NOOP for small talk ==========
    {
        "memory": "[m1] User likes hiking\n[m2] User lives in Colorado",
        "event": "User says: Good morning! How are you today?",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User is a data scientist\n[m2] User uses Python",
        "event": "User says: Hmm, let me think about that for a moment.",
        "operation": "NOOP"
    },
    {
        "memory": "[m1] User has a dog named Buddy\n[m2] User walks Buddy twice a day",
        "event": "User says: Yeah that makes sense, thanks for explaining.",
        "operation": "NOOP"
    },
    # ========== Additional diverse ADD ==========
    {
        "memory": "(empty memory)",
        "event": "User says: I'm color blind - specifically red-green color blindness. Please avoid color-dependent explanations.",
        "operation": "ADD: User has red-green color blindness - avoid color-dependent explanations."
    },
    {
        "memory": "[m1] User is a musician\n[m2] User plays violin",
        "event": "User says: I just joined the city orchestra as first violinist!",
        "operation": "ADD: User is first violinist in the city orchestra."
    },
    {
        "memory": "[m1] User has 3 children\n[m2] User lives in a 3-bedroom house",
        "event": "User says: Surprise - we're expecting twins! Need a bigger house now.",
        "operation": "ADD: User is expecting twins, needs to find a bigger house."
    },
    {
        "memory": "[m1] User studies machine learning\n[m2] User is in their first year of PhD",
        "event": "User says: I'm switching my research focus from computer vision to large language models.",
        "operation": "UPDATE m1: User's research focus changed from CV to large language models."
    },
    {
        "memory": "[m1] User is a vegan\n[m2] User cooks at home daily",
        "event": "User says: My nutritionist recommended adding eggs to my diet for protein.",
        "operation": "UPDATE m1: User is no longer fully vegan - added eggs per nutritionist advice."
    },
    {
        "memory": "[m1] User is building a mobile app for fitness tracking",
        "event": "User says: The app just hit 10,000 downloads on the App Store!",
        "operation": "ADD: User's fitness app reached 10,000 downloads on App Store."
    },
    {
        "memory": "[m1] User's rent is $2000/month\n[m2] User lives downtown",
        "event": "User says: The landlord raised our rent to $2500 starting next month.",
        "operation": "UPDATE m1: User's rent is increasing to $2500/month."
    },
    {
        "memory": "[m1] User flies economy class\n[m2] User travels for work monthly",
        "event": "User says: Company upgraded our travel policy - business class for flights over 6 hours!",
        "operation": "UPDATE m1: User now flies business class for flights over 6 hours (company policy)."
    },
]


def generate_sft_data() -> list[dict]:
    """Generate 200+ diverse SFT training examples for Memory Manager.

    Returns formatted prompt-completion pairs for supervised fine-tuning.
    """
    formatted = []
    for ex in _SFT_RAW_EXAMPLES:
        prompt = (
            "You are a Memory Manager for an AI agent. "
            "Given the current memory entries and a new event, "
            "decide the best memory operation.\n\n"
            "### Available Operations\n"
            "- ADD: <content> — Store new important information\n"
            "- UPDATE <id>: <new_content> — Update existing memory\n"
            "- DELETE <id> — Remove outdated/wrong memory\n"
            "- NOOP — No action needed\n\n"
            f"### Current Memory Entries\n{ex['memory']}\n\n"
            f"### New Event\n{ex['event']}\n\n"
            "### Decision\n"
        )
        formatted.append({
            "text": prompt + ex["operation"],
            "prompt": prompt,
            "completion": ex["operation"],
        })

    logger.info(f"Generated {len(formatted)} unique SFT examples")
    return formatted


def main(args):
    set_seed(42)
    logger.info(f"Phase 0: SFT Warmup | model={args.model_name}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
    )

    # Setup LoRA for efficient SFT
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Generate training data
    train_data = generate_sft_data()
    logger.info(f"Total SFT examples: {len(train_data)}")

    # Shuffle data
    random.shuffle(train_data)

    # SFT training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_data)
    )
    model.train()

    best_loss = float("inf")
    for epoch in range(args.num_epochs):
        total_loss = 0
        random.shuffle(train_data)  # Re-shuffle each epoch
        for i, example in enumerate(train_data):
            inputs = tokenizer(
                example["text"], return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} | avg_loss={avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(args.output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))

    logger.info(f"Phase 0 complete. Best loss={best_loss:.4f}, saved to {args.output_dir}/best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Phase 0: SFT Warmup")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="outputs/phase0")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    main(args)
