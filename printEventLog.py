from tensorflow.python.summary.summary_iterator import summary_iterator
import yaml

direct = '/data/alex/PHAL/runs/PHAL_3/logs/events.out.tfevents.1648048044.frodo.1625602.0' #'/data/alex/original_torch/runs/elu_fx_original_.1.1.2.2.3DO_padSame_dice_sigmoid/logs/events.out.tfevents.1647372423.frodo.4062178.0'
for summary in summary_iterator(direct): #/data/alex/original_torch/runs/0_elu_fx_gaussianNeighbor_original_.1.1.2.2.3DropOut_padSame_noReg/logs/events.out.tfevents.1646849487.frodo.3757538.0'):
    for v in summary.summary.value:
        if v.tag=='Configuration Parameters/text_summary':
            for s in v.tensor.string_val:
                d = yaml.safe_load(str(s,encoding='utf-8', errors='strict'))
                for key, value in d.items():
                    print(key + " : " + str(value))

'''n=0
for summary in summary_iterator(direct):
    for v in summary.summary.value:
        print(v)
    n+=1
    if n==14: break'''
    