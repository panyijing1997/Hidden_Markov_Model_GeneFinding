from training_by_counting import *

model = load_model('models/validated_on_3')

for i in range(9, 11):
    print("Genome " + str(i) + " predicting...")
    genome = read_fasta_file('data/genome' + str(i) + '.fa')["genome" + str(i)]

    pred_ann = get_ann(model, genome)

    with open('prediction/pred-ann' + str(i) + '.fa', 'w') as f:
        prefix = ">pred-ann" + str(i) + "\n"
        f.write(prefix + pred_ann)
    print("Genome " + str(i) + " done")