#!/usr/bin/env python3

import os
import json
import argparse
import torch
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


def load_jsonl(path):

    rows=[]

    with open(path,encoding="utf8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows



def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--student")
    parser.add_argument("--teacher")
    parser.add_argument("--data")
    parser.add_argument("--output")

    parser.add_argument("--epochs",
                        type=int,
                        default=3)

    parser.add_argument("--lr",
                        type=float,
                        default=1e-6)

    args=parser.parse_args()


    rank=int(os.environ.get("RANK",0))

    world=int(
        os.environ.get(
            "WORLD_SIZE",
            1
        )
    )


    if world>1:

        dist.init_process_group(
            backend="hccl"
        )


    device=f"npu:{rank}"


    if rank==0:

        print("="*70)
        print("PE CONDITIONED REVERSE KL OPD")
        print("="*70)



    tokenizer=AutoTokenizer.from_pretrained(
        args.student,
        local_files_only=True
    )


    student=AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)



    teacher=AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)



    teacher.eval()


    for p in teacher.parameters():
        p.requires_grad=False



    rows=load_jsonl(args.data)


    if rank==0:
        print(
            "ROWS=",
            len(rows)
        )


    optimizer=torch.optim.AdamW(
        student.parameters(),
        lr=args.lr
    )


    student.train()


    step=0


    for epoch in range(args.epochs):

        for item in rows:

            source=item.get("source","")

            draft=item.get(
                "student_translation",
                ""
            )

            target=item.get(
                "corrected_translation",
                ""
            )


            prompt=f"""
Translate the following sentence.

Source:
{source}

Previous translation:
{draft}

Correct translation:
"""


            text=prompt+target


            inputs=tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)


            labels=inputs.input_ids.clone()



            with torch.no_grad():

                teacher_out=teacher(
                    **inputs
                )

                teacher_logits=(
                    teacher_out.logits
                )


            student_out=student(
                **inputs
            )


            student_logits=(
                student_out.logits
            )


            # reverse KL:
            #
            # KL(Student || Teacher)

            s_logp=torch.log_softmax(
                student_logits,
                dim=-1
            )


            t_prob=torch.softmax(
                teacher_logits,
                dim=-1
            )


            loss=torch.sum(
                t_prob*
                (
                    torch.log(t_prob+1e-8)
                    -
                    s_logp
                ),
                dim=-1
            ).mean()



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            step+=1


            if rank==0 and step%50==0:

                print(
                    {
                    "epoch":epoch,
                    "step":step,
                    "loss":float(loss)
                    },
                    flush=True
                )



    if rank==0:

        os.makedirs(
            args.output,
            exist_ok=True
        )

        student.save_pretrained(
            args.output
        )

        tokenizer.save_pretrained(
            args.output
        )


        print(
            "PE_OPD_REVERSEKL_FINISH"
        )



if __name__=="__main__":
    main()

