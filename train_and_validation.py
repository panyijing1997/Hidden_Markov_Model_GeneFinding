from training_by_counting import *
if __name__ == '__main__':
    for i in range (1,6):
        print("training by counting round "+str(i))
        model = training_for_gene_finding(i)
        save_model(model, 'models/validated_on_'+str(i))
        model = load_model('models/validated_on_'+str(i))
        print("init probs")
        print(model.init_probs)
        print("trans probs")
        print(model.trans_probs)
        print("emission probs")
        print(model.emission_probs)


    for i in range(1,6):

        print("Round "+str(i)+" predicting...")
        genome = read_fasta_file('data/genome'+str(i)+'.fa')["genome"+str(i)]
        model = load_model('models/validated_on_' + str(i))
        pred_ann = get_ann(model, genome )

        with open('validation/pred-ann'+str(i)+'.fa', 'w') as f:
            prefix=">pred-ann" + str(i)+"\n"
            f.write(prefix+pred_ann)
        print("Round "+str(i)+" done")


