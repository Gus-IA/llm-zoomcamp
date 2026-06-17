Q1 = "I just discovcered the course, can I still join?"
Q2 = "I just found out abaout the program, can I still enroll?"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

v1 = model.encode(Q1)

print(v1.shape)


q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)

d = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
dv = model.encode(d)

print(v1.dot(dv))

q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)

v2.dot(dv)

print(v2.dot(dv))
