Code for our ICML-2020 paper [**Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation**](https://arxiv.org/abs/2002.08546). 

### Framework:  

<img src="figs/shot.jpg" width="600"/>

### Results:

#### Table 2 [UDA results on Digits]

| Methods    | S->M | U->M | M->U | Avg. |
| -------------- | ---- | ---- | ---- | ---- |
| srconly (2019) | 71.5 | 85.5 | 82.5 |  	 |
| srconly (2020) | 69.2 | 89.8 | 77.6 |   	 |
| srconly (2021) | 69.7 | 88.7 | 79.0 |  	 |
| srconly (Avg.) | 70.2 | 88.0 | 79.7 | 79.3 |
| SHOT-IM (2019) | 98.9 | 98.6 | 97.8 |  	 |
| SHOT-IM (2020) | 99.0 | 97.8 | 97.7 |  	 |
| SHOT-IM (2021) | 98.9 | 97.6 | 97.7 |  	 |
| SHOT-IM (Avg.) | 99.0 | 97.6 | 97.7 | 98.2 |
| SHOT (2019)    | 98.8 | 98.6 | 98.0 |   	 |
| SHOT (2020)    | 99.0 | 97.6 | 97.8 |  	 |
| SHOT (2021)    | 99.0 | 97.7 | 97.7 |  	 |
| SHOT (Avg.)    | 98.9 | 98.0 | 97.9 | 98.3 |
| Oracle (2019)  | 99.2 | 99.2 | 97.1 |  	 |
| Oracle (2020)  | 99.2 | 99.2 | 97.0 |  	 |
| Oracle (2021)  | 99.3 | 99.3 | 97.0 |  	 |
| Oracle (Avg.)  | 99.2 | 99.2 | 97.0 | 98.8 |

#### Table 2 [UDA results on Digits] (Vinicius Arruda replication + experiments)


| Methods    | S->M | U->M | M->U | Avg. |
| -------------- | ---- | ---- | ---- | ---- |
| srconly (2019) | 74.26 | 85.87 | 77.04 |  	 |
| srconly (2020) | 68.50 | 87.37 | 84.30 |   	 |
| srconly (2021) | 72.49 | 89.09 | 77.96 |  	 |
| srconly (Avg.) | 71.75 | 87.44 | 79.77 | 79.65 |
| SHOT-ENT (2020) | 99.04 | 97.68 | 97.80 |  	 |
| SHOT-GENT (2020) | 21.30 | 29.38 | 23.66 |  	 |
| SHOT-IM-NO-SRC-SMOOTH (2020) | |  | 98.12 |  	 |  (from srconly 71.67 m2u)
| SHOT-IM-NO-H-FREEZE (2020) | |  | 97.80 |  	 |  (from srconly 77.37 m2u)
| SHOT-IM-NO-SRC-SMOOTH-H-FREEZE (2020) | |  | 98.33 |  	 |  (from srconly 77.37 m2u)
| SHOT-IM (2019) | 99.12 | 97.57 | 97.80 |  	 |
| SHOT-IM (2020) | 99.02 | 97.71 | 98.23 |  	 |
| SHOT-IM (2021) | 98.77 | 98.83 | 97.85 |  	 |
| SHOT-IM (Avg.) | 98.97 | 98.04 | 97.96 | 98.32 |
| SHOT-H-FREEZE (2020) | |  | 97.96 |  	 |  (from srconly 77.37 m2u)
| SHOT-NO-SRC-SMOOTH-H-FREEZE (2020) | |  | 97.74 |  	 |  (from srconly 77.37 m2u)
| SHOT (2019)    | 98.95 | 97.63 | 97.80 |   	 |
| SHOT (2020)    | 98.00 | 97.72 | 98.17 |  	 |
| SHOT (2021)    | 98.93 | 98.95 | 97.90 |  	 |
| SHOT (Avg.)    | 98.63 | 98.10 | 97.96 | 98.23 |
| Oracle (2019)  |  |  |  |  	 |
| Oracle (2020)  |  |  |  |  	 |
| Oracle (2021)  |  |  |  |  	 |
| Oracle (Avg.)  |  |  |  |  |

mnist -> usps
refazer.. conferir.. fazer desligando e ligando cada coisa! deixar codigo pronto e colocar os experimentos para rodar..
olhar o todo la no tasks

#### Table 3 [UDA results on Office]

| Methods        | A->D | A->W | D->A | D->W | W->A | W->D | Avg. |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| srconly (2019) | 79.9 | 77.5 | 58.9 | 95.0 | 64.6 | 98.4 |      |
| srconly (2020) | 81.5 | 75.8 | 61.6 | 96.0 | 63.3 | 99.0 |      |
| srconly (2021) | 80.9 | 77.5 | 60.2 | 94.8 | 62.9 | 98.8 |      |
| srconly (Avg.) | 80.8 | 76.9 | 60.3 | 95.3 | 63.6 | 98.7 | 79.3 |
| SHOT-IM (2019) | 88.8 | 90.7 | 71.7 | 98.5 | 71.7 | 99.8 |      |
| SHOT-IM (2020) | 92.6 | 92.2 | 72.4 | 98.4 | 71.1 | 100. |      |
| SHOT-IM (2021) | 90.6 | 90.7 | 73.3 | 98.0 | 71.2 | 99.8 |      |
| SHOT-IM (Avg.) | 90.6 | 91.2 | 72.5 | 98.3 | 71.4 | 99.9 | 87.3 |
| SHOT (2019)    | 93.4 | 88.8 | 74.9 | 98.5 | 74.6 | 99.8 |      |
| SHOT (2020)    | 95.0 | 92.0 | 75.7 | 98.6 | 73.7 | 100. |      |
| SHOT (2021)    | 93.8 | 89.7 | 73.6 | 98.2 | 74.6 | 99.8 |      |
| SHOT (Avg.)    | 94.0 | 90.1 | 74.7 | 98.4 | 74.3 | 99.9 | 88.6 |

#### Table 4 [UDA results on Office-Home]

| Methods        |Ar->Cl|Ar->Pr|Ar->Re|Cl->Ar|Cl->Pr|Cl->Re|Pr->Ar|Pr->Cl|Pr->Re|Re->Ar|Re->Cl|Re->Pr| Avg. |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| srconly (2019) | 45.2 | 67.2 | 75.0 | 52.4 | 62.6 | 64.9 | 52.4 | 40.6 | 73.0 | 65.0 | 43.8 | 78.1 |   |
| srconly (2020) | 44.2 | 67.2 | 74.4 | 52.3 | 63.1 | 64.5 | 53.1 | 41.0 | 73.7 | 65.3 | 46.8 | 77.9 |   |
| srconly (2021) | 44.5 | 67.7 | 74.8 | 53.4 | 62.4 | 64.9 | 53.4 | 40.4 | 72.9 | 65.7 | 45.8 | 78.1 |   |
| srconly (Avg.) | 44.6 | 67.3 | 74.8 | 52.7 | 62.7 | 64.8 | 53.0 | 40.6 | 73.2 | 65.3 | 45.4 | 78.0 | 60.2 |
| SHOT-IM (2019) | 56.5 | 77.1 | 80.8 | 67.7 | 73.3 | 75.1 | 65.5 | 54.5 | 80.6 | 73.4 | 57.2 | 84.0 |   |
| SHOT-IM (2020) | 54.7 | 76.3 | 80.2 | 66.8 | 75.8 | 76.2 | 65.6 | 53.9 | 80.7 | 73.6 | 58.3 | 83.5 |   |
| SHOT-IM (2021) | 54.9 | 76.4 | 80.1 | 66.2 | 73.8 | 75.0 | 65.7 | 56.1 | 80.7 | 74.2 | 59.6 | 82.9 |   |
| SHOT-IM (Avg.) | 55.4 | 76.6 | 80.4 | 66.9 | 74.3 | 75.4 | 65.6 | 54.8 | 80.7 | 73.7 | 58.4 | 83.4 | 70.5 |
| SHOT (2019)    | 57.3 | 79.3 | 81.8 | 68.1 | 77.1 | 78.0 | 67.8 | 55.0 | 82.5 | 73.2 | 58.5 | 84.1 |   |
| SHOT (2020)    | 57.1 | 77.5 | 81.6 | 68.4 | 78.2 | 77.9 | 67.0 | 55.6 | 82.4 | 73.6 | 60.2 | 84.6 |   |
| SHOT (2021)    | 57.0 | 77.6 | 81.0 | 67.5 | 79.2 | 78.3 | 67.3 | 54.1 | 81.6 | 73.0 | 57.8 | 84.2 |   |
| SHOT (Avg.)    | 57.1 | 78.1 | 81.5 | 68.0 | 78.2 | 78.1 | 67.4 | 54.9 | 82.2 | 73.3 | 58.8 | 84.3 | 71.8 |

#### Table 5 [UDA results on VisDA-C]

| Methods        | plane | bcycl | bus | car | horse | knife | mcycl | person | plant | sktbrd | train | truck | Per-class |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| srconly (2019) | 57.1 | 20.5 | 48.6 | 60.8 | 66.2 | 3.6 | 80.7 | 23.9 | 38.5 | 31.0 | 87.0 | 10.7 |   |
| srconly (2020) | 65.1 | 18.9 | 57.2 | 66.9 | 69.9 | 11.0| 84.7 | 23.9 | 69.4 | 34.0 | 83.8 | 9.3 |   |
| srconly (2021) | 60.5 | 25.5 | 47.0 | 75.2 | 61.3 | 4.2 | 81.1 | 21.9 | 63.9 | 26.7 | 83.1 | 4.0 |   |
| srconly (Avg.) | 60.9 | 21.6 | 50.9 | 67.6 | 65.8 | 6.3 | 82.2 | 23.2 | 57.3 | 30.6 | 84.6 | 8.0 | 46.6 |
| SHOT-IM (2019) | 94.3 | 86.6 | 78.1 | 54.0 | 91.0 | 92.3 | 79.1 | 78.9 | 88.4 | 86.0 | 88.0 | 50.7 |   |
| SHOT-IM (2020) | 93.4 | 87.1 | 80.4 | 51.7 | 91.5 | 92.9 | 80.0 | 78.0 | 89.6 | 85.1 | 87.2 | 51.3 |    |
| SHOT-IM (2021) | 93.5 | 85.7 | 77.6 | 46.3 | 90.5 | 95.1 | 77.9 | 78.1 | 89.7 | 85.0 | 88.5 | 51.2 |    |
| SHOT-IM (Avg.) | 93.7 | 86.4 | 78.7 | 50.7 | 91.0 | 93.5 | 79.0 | 78.3 | 89.2 | 85.4 | 87.9 | 51.1 | 80.4 |
| SHOT (2019)    | 93.8 | 89.0 | 81.4 | 57.0 | 93.4 | 94.7 | 81.3 | 80.3 | 90.5 | 89.1 | 85.3 | 58.4 |   |
| SHOT (2020)    | 94.5 | 87.3 | 80.0 | 57.1 | 93.1 | 94.5 | 82.0 | 80.7 | 91.7 | 89.4 | 87.0 | 58.3 |   |
| SHOT (2021)    | 94.7 | 89.1 | 78.7 | 57.8 | 92.8 | 95.5 | 78.8 | 79.9 | 92.4 | 89.0 | 86.6 | 57.9 |   |
| SHOT (Avg.)    | 94.3 | 88.5 | 80.1 | 57.3 | 93.1 | 94.9 | 80.7 | 80.3 | 91.5 | 89.1 | 86.3 | 58.2 | 82.9 |

#### Table 7 [PDA/ ODA results on Office-Home]

| Methods@PDA       |Ar->Cl|Ar->Pr|Ar->Re|Cl->Ar|Cl->Pr|Cl->Re|Pr->Ar|Pr->Cl|Pr->Re|Re->Ar|Re->Cl|Re->Pr| Avg. |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| srconly (2019) | 46.0 | 69.7 | 80.7 | 56.3 | 60.4 | 66.9 | 60.2 | 40.6 | 76.0 | 70.8 | 48.6 | 78.5 |   |
| srconly (2020) | 45.1 | 71.0 | 80.8 | 55.7 | 61.8 | 66.4 | 61.4 | 39.7 | 76.1 | 70.6 | 49.7 | 76.3 |   |
| srconly (2021) | 44.5 | 70.5 | 81.3 | 56.8 | 60.2 | 65.2 | 61.2 | 40.0 | 76.5 | 70.9 | 47.2 | 77.2 |   |
| srconly (Avg.) | 45.2 | 70.4 | 81.0 | 56.2 | 60.8 | 66.2 | 60.9 | 40.1 | 76.2 | 70.8 | 48.5 | 77.3 | 62.8 |
| SHOT-IM (2019) | 57.5 | 86.2 | 88.2 | 69.3 | 73.6 | 79.9 | 79.7 | 62.2 | 89.0 | 80.8 | 66.6 | 91.0 |   |
| SHOT-IM (2020) | 61.2 | 82.0 | 87.8 | 73.3 | 74.4 | 80.6 | 74.1 | 58.8 | 90.0 | 81.7 | 70.8 | 87.1 |   |
| SHOT-IM (2021) | 55.0 | 82.6 | 90.3 | 74.5 | 74.0 | 76.5 | 74.4 | 60.8 | 91.4 | 83.0 | 67.5 | 87.3 |   |
| SHOT-IM (Avg.) | 57.9 | 83.6 | 88.8 | 72.4 | 74.0 | 79.0 | 76.1 | 60.6 | 90.1 | 81.9 | 68.3 | 88.5 | 76.8 |
| SHOT (2019)    | 65.0 | 85.0 | 93.3 | 75.7 | 79.3 | 88.9 | 80.5 | 65.3 | 90.1 | 80.9 | 67.0 | 86.3 |   |
| SHOT (2020)    | 64.1 | 82.0 | 92.7 | 77.6 | 74.8 | 90.7 | 80.0 | 63.5 | 88.4 | 79.9 | 66.8 | 85.0 |   |
| SHOT (2021)    | 65.2 | 88.7 | 92.2 | 75.7 | 78.8 | 86.8 | 78.5 | 64.1 | 90.1 | 80.9 | 65.3 | 86.0 |   |
| SHOT (Avg.)    | 64.8 | 85.2 | 92.7 | 76.3 | 77.6 | 88.8 | 79.7 | 64.3 | 89.5 | 80.6 | 66.4 | 85.8 | 79.3 |


| Methods@ODA        |Ar->Cl|Ar->Pr|Ar->Re|Cl->Ar|Cl->Pr|Cl->Re|Pr->Ar|Pr->Cl|Pr->Re|Re->Ar|Re->Cl|Re->Pr| Avg. |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| srconly (2019) | 37.4 | 54.7 | 69.9 | 34.2 | 44.3 | 49.7 | 37.7 | 30.1 | 56.2 | 50.6 | 35.2 | 61.6 |   |
| srconly (2020) | 36.4 | 55.0 | 69.0 | 33.3 | 44.7 | 47.8 | 34.6 | 29.2 | 55.7 | 53.2 | 36.0 | 62.4 |   |
| srconly (2021) | 35.1 | 54.8 | 68.4 | 33.8 | 44.1 | 50.1 | 38.2 | 28.1 | 58.3 | 50.3 | 34.1 | 62.9 |   |
| srconly (Avg.) | 36.3 | 54.8 | 69.1 | 33.8 | 44.4 | 49.2 | 36.8 | 29.2 | 56.8 | 51.4 | 35.1 | 62.3 | 46.6 |
| SHOT-IM (2019) | 61.6 | 80.1 | 84.4 | 61.8 | 74.0 | 81.9 | 63.6 | 58.5 | 83.1 | 68.4 | 63.7 | 82.2 |   |
| SHOT-IM (2020) | 63.4 | 76.0 | 83.2 | 61.4 | 74.3 | 78.7 | 63.8 | 59.6 | 83.1 | 70.0 | 61.8 | 82.7 |   |
| SHOT-IM (2021) | 62.4 | 77.3 | 84.1 | 59.6 | 71.9 | 77.7 | 66.7 | 58.0 | 83.0 | 68.9 | 60.6 | 81.6 |   |
| SHOT-IM (Avg.) | 62.5 | 77.8 | 83.9 | 60.9 | 73.4 | 79.4 | 64.7 | 58.7 | 83.1 | 69.1 | 62.0 | 82.1 | 71.5 |
| SHOT (2019)    | 63.9 | 80.6 | 85.6 | 63.6 | 77.1 | 83.2 | 64.9 | 58.3 | 83.2 | 69.7 | 65.2 | 82.8 |   |
| SHOT (2020)    | 64.0 | 80.4 | 84.7 | 63.4 | 75.3 | 81.6 | 65.1 | 60.9 | 82.8 | 69.9 | 64.4 | 82.4 |   |
| SHOT (2021)    | 65.6 | 80.2 | 83.8 | 62.2 | 73.7 | 78.8 | 65.9 | 58.8 | 83.9 | 69.2 | 64.1 | 81.7 |   |
| SHOT (Avg.)    | 64.5 | 80.4 | 84.7 | 63.1 | 75.4 | 81.2 | 65.3 | 59.3 | 83.3 | 69.6 | 64.6 | 82.3 | 72.8 |

#### Table 8 [MSDA/ MTDA results on Office-Caltech]

| Methods@MSDA   | ->Ar | ->Cl | ->Pr | ->Re | Avg. |
| -------------- | ---- | ---- | ---- | ---- | ---- |
| srconly (2019) | 95.2 | 93.9 | 99.4 | 98.0 | |
| srconly (2020) | 95.4 | 93.5 | 98.7 | 98.6 | |
| srconly (2021) | 95.6 | 93.7 | 98.7 | 98.3 | |
| srconly (Avg.) | 95.4 | 93.7 | 98.9 | 98.3 | 96.6 |
| SHOT-IM (2019) | 95.8 | 96.0 | 99.4 | 99.7 | |
| SHOT-IM (2020) | 96.5 | 95.9 | 97.5 | 99.7 | |
| SHOT-IM (2021) | 96.4 | 96.3 | 98.7 | 99.7 | |
| SHOT-IM (Avg.) | 96.2 | 96.1 | 98.5 | 99.7 | 97.6 |
| SHOT (2019) | 96.2 | 95.9 | 98.7 | 99.7 | |
| SHOT (2020) | 96.5 | 96.1 | 98.7 | 99.7 |
| SHOT (2021) | 96.6 | 96.6 | 98.1 | 99.7 | |
| SHOT (Avg.) | 96.4 | 96.2 | 98.5 | 99.7 | 97.7 |

| Methods@MTDA   | Ar-> | Cl-> | Pr-> | Re-> | Avg. |
| -------------- | ---- | ---- | ---- | ---- | ---- |
| srconly (2019) | 90.4 | 95.9 | 90.3 |	90.6 | |
| srconly (2020) | 91.2 | 95.9 | 90.2 | 91.1 | |
| srconly (2021) | 90.5 | 96.5 | 90.2 | 91.1 | |
| srconly (Avg.) | 90.7 | 96.1 | 90.2 | 90.9 | 92.0 |
| SHOT-IM (2019) | 96.6 | 97.5 | 96.3 | 96.0 | |
| SHOT-IM (2020) | 95.1 | 96.7 | 96.3 | 96.4 | |
| SHOT-IM (2021) | 95.4 | 97.3 | 96.3 | 96.0 | |
| SHOT-IM (Avg.) | 95.7 | 97.2 | 96.3 | 96.1 | 96.3 |
| SHOT (2019) | 96.6 | 97.5 | 96.4 | 96.0 | |
| SHOT (2020) | 95.4 | 97.0 | 96.5 | 96.7 | |
| SHOT (2021) | 96.6 | 97.5 | 96.0 | 96.0 | |
| SHOT (Avg.) | 96.2 | 97.3 | 96.3 | 96.2 | 96.5 |


#### Table 9 [PDA results on ImageNet->Caltech]

| Methods@PDA    | 2019 | 2020 | 2021 | Avg. |
| -------------- | ---- | ---- | ---- | ---- |
| srconly 		 | 69.7 | 69.7 | 69.7 | 69.7 |
| SHOT-IM		 | 81.1 | 82.2 | 81.8 | 81.7 |
| SHOT			 | 83.2 | 83.3 | 83.4 | 83.3 |




### Citation

If you find this code useful for your research, please cite our paper

> @inproceedings{liang2020shot,  
>  &nbsp; &nbsp;  title={Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation},  
>  &nbsp; &nbsp;  author={Liang, Jian and Hu, Dapeng and Feng, Jiashi},  
>  &nbsp; &nbsp;  booktitle={International Conference on Machine Learning (ICML)},  
>  &nbsp; &nbsp;  pages={xx-xx},  
>  &nbsp; &nbsp;  month = {July},  
>  &nbsp; &nbsp;  year={2020}  
> }

### Contact

- [liangjian92@gmail.com](mailto:liangjian92@gmail.com)
- [dapeng.hu@u.nus.edu](mailto:dapeng.hu@u.nus.edu)
- [elefjia@nus.edu.sg](mailto:elefjia@nus.edu.sg)
