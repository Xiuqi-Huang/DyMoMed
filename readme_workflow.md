- base_doctor.py 用来运行对比实验的医生agent的模型。
- base_doctor_dagent.py 用来运行DoctorAgent-RL的医生agent的模型。
- patient.py 病人agent和病人监测器agent的模型。
- interaction.py 医患交互的流程代码。
    运行时需要关注这里的路径是否和本地适配result_dir = f"../result/inter/{args.agent_dataset}"；
    填写使用的apiparser.add_argument('--doctor_base_url', type=str, default='to be filled')；
    填写使用的apiparser.add_argument('--patient_base_url', type=str, default='to be filled')。
- evaluation.py 评估医生agent的代码。
    运行时需要关注这里的路径是否和本地适配out_dir = f"../result/eval/{args.dataset}"。
- utils.py 存放各种基础函数。
    运行时需要关注这里的路径是否和本地适配DATA_MAP = {
            'pmc': "../data/pmc100.json",
            'clinic': "../data/clinic100.json",
            'mtmed': "../data/mtmed100.json"}。
- run_inter.sh 运行交互流程的代码。
- run_eval.sh 运行评估流程的代码。

