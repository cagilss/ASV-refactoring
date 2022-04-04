from operator import index
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import bob.measure as bm
import numpy as np
import os
import math 
from sklearn.metrics import roc_curve, precision_recall_curve, auc



class Evaluation_EER:
    
    #def __init__(self):
        #self.train_logits, self.train_softmax_logits, self.train_true_labels, self.train_sigmoid = self.load('train')
        #self.dev_logits, self.dev_softmax_logits, self.dev_true_labels, self.dev_sigmoid = self.load('dev')
        #self.eval_logits, self.eval_softmax_logits, self.eval_true_labels, self.eval_sigmoid = self.load('eval')


    def negatives(self, data, label, spoof=0):
        List = []
        for idx,L in enumerate(label):
            if L == spoof:
                score = data[idx].tolist()
                List.append(score) 
        return List
    
    def positives(self, data, label, genuine=1):
        List = []
        for idx,L in enumerate(label):
            if L == genuine:
                score = data[idx].tolist()
                List.append(score) 
        return List
    
    def compute_eer(self, data, label):
        EER_list = []
        for D,L in zip(data, label):
            scores = D[:, 1] - D[:, 0]
            neg = self.negatives(scores, L) 
            pos = self.positives(scores, L) 
            try:
                EER_list.append(bm.eer_rocch(neg, pos))
            except RuntimeError:
                pass
        return np.mean(EER_list)


    def miss_classified_samples(self, logits, labels):
        pred = np.argmax(logits, axis=1)
        miss_idx = np.where(pred != labels)
        miss_cls = logits[miss_idx]
        return miss_cls


    def true_classified_samples(self, logits, labels):
        pred = np.argmax(logits, axis=1)
        true_idx = np.where(pred == labels)
        true_cls = logits[true_idx]
        return true_cls

    def true_classified_samples_1d(self, logits, labels, thr=0.5):
        pred = [1 if L >= thr else 0 for L in logits]
        true_idx = np.where(pred == labels)
        true_cls = logits[true_idx]
        return true_cls

    def miss_classified_samples_1d(self, logits, labels, thr=0.5):
        pred = [1 if L >= thr else 0 for L in logits]
        true_idx = np.where(pred != labels)
        true_cls = logits[true_idx]
        true_lbl = labels[true_idx]
        return true_cls, true_lbl


    def eer(self):
        train_eer = self.compute_eer(self.train_logits, self.train_true_labels)
        dev_eer   = self.compute_eer(self.dev_logits, self.dev_true_labels)
        eval_eer  = self.compute_eer(self.eval_logits, self.eval_true_labels)
        print('train eer: {} dev eer:{} eval eer: {}'.format(train_eer, dev_eer, eval_eer))



    def load(self, typ, e): 
        #folder = 'scores'

        # LCNN model scores 
        #folder = 'scores_lcnn_dynamic_pad/scores_tr_dev_norm_cr_en' # 47
        #folder = 'scores_lcnn_dynamic_pad/scores_tr_dev_norm_edl' # 43

        #folder = 'scores_lcnn_dynamic_pad/lcnn_edl_and_cr_en_softmax_and_probs/scores_cr_en' # 47
        folder = 'scores_0776eer_103' # 43

        # ResNet model scores 
        #folder = 'S8_ce_resnet_1101_60'
        #folder = 'S8_edl_renet_1054_58' # 33

        # LSTM model scores 
        #folder = 'scores_lstm_dynamic_pad/scores_tr_dev_norm_cr_en_lstm' # 48
        #folder = 'scores_lstm_dynamic_pad/scores_tr_dev_norm_edl_lstm' # 43

        logit         = '{}_logit_scores_{}.npy'.format(typ, e)
        sigmoid       = '{}_sigmoid_scores_{}.npy'.format(typ, e)
        softmax_logit = '{}_softmax_scores_{}.npy'.format(typ, e)
        probs         = '{}_probs_scores_{}.npy'.format(typ, e)
        true_label    = '{}_labels_{}.npy'.format(typ, e)
        
        p1 = os.path.join(folder, logit)
        p2 = os.path.join(folder, softmax_logit)
        p3 = os.path.join(folder, true_label)
        p4 = os.path.join(folder, sigmoid)
        p5 = os.path.join(folder, probs)
        return self.load_(p1), self.load_(p2), self.load_(p3), self.load_(p4), self.load_(p5)


    def load_(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        else:
            return np.load(path, allow_pickle=True)


    def roc_curve_(self, normal, anormal): # normal: not match, anormal: match
        truth = np.ones((normal.shape[0]+anormal.shape[0],))
        truth[normal.shape[0]:]=0
        x = np.concatenate((normal,anormal),0)
        fpr, tpr, _ = roc_curve(truth, x, 0)
        roc_auc = auc(fpr, tpr)             
        return fpr, tpr, roc_auc


    """ plots the roc curve by using bob.measure library """
    def compute_roc_curve_bob(self, logits=None, labels=None, write_as_txt=False, typ=None, model_typ=None):
        positives = list()
        negatives = list()

        # seperate logits as positive and negatives
        for batch, label in zip(logits, labels): # shape (32, 2)
            for x, y in zip(batch, label):
                s = x[1] - x[0]
                if y == 1: # genuine
                    positives.append(s)
                else: # spoof 
                    negatives.append(s)
                
        # convert float32 to float64
        negatives = np.double(negatives)
        positives = np.double(positives)
        
        # compute tpr and fpr
        fpr, fnr = bm.roc(negatives, positives, n_points=2000)
        tpr = 1 - fnr

        # compute AUC
        from numpy import trapz
        tpr_s = np.sort(tpr)
        fpr_s = np.sort(fpr)

        area = trapz(tpr_s, x=fpr_s)
        print(area)

        # plot roc curve 
        plt.plot(fpr, tpr, 'r--') 
        plt.grid(True)
        #plt.show()


    def compute_roc_curve(self, logits=None, labels=None, write_as_txt=False, typ=None, model_typ=None): 
        L_match, L_not_match = [], []

        if logits is None and labels is None: 
            logits = self.eval_logits
            labels = self.eval_true_labels

        for x,y in zip(logits, labels):
            match = self.true_classified_samples(x, y)
            not_match = self.miss_classified_samples(x, y)
            for v in match.max(1).tolist(): L_match.append(v)
            for v in not_match.max(1).tolist(): L_not_match.append(v)

        L_not_match = np.array(L_not_match)
        L_match = np.array(L_match)
        fpr, tpr, area = self.roc_curve_(L_not_match, L_match)

        print('AUC:', area)

        if write_as_txt:
            with open('scores_roc/{}_negatives_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in fpr:
                    f.write(str(line) + '\n')

            with open('scores/roc/{}_positives_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in tpr:
                    f.write(str(line) + '\n')
    
    def load_scores(self, model_typ, tf, typ, folder_typ='cdf'):
        with open('scores_{}/{}_{}_x_{}.txt'.format(folder_typ, model_typ, tf, typ), 'r') as f:
            x = [float(line.strip()) for line in f.readlines()]

        with open('scores_{}/{}_{}_y_{}.txt'.format(folder_typ, model_typ, tf, typ), 'r') as f:
            y = [float(line.strip()) for line in f.readlines()]
        
        return x, y


    def plot_roc_curve(self):
        #lcnn_x_edl, lcnn_y_edl = self.load_scores(model_typ='lcnn', typ='edl')
        #lcnn_x_ce, lcnn_y_ce = self.load_scores(model_typ='lcnn', typ='ce') 

        resnet_x_edl, resnet_y_edl = self.load_scores(model_typ='resnet', typ='edl')
        resnet_x_ce, resnet_y_ce = self.load_scores(model_typ='resnet', typ='ce')

        lstm_x_edl, lstm_y_edl = self.load_scores(model_typ='lstm', typ='edl')
        lstm_x_ce, lstm_y_ce = self.load_scores(model_typ='lstm', typ='ce')

        lcnn_area_ce = round(auc(lcnn_x_ce, lcnn_y_ce), 2)
        lcnn_area_edl = round(auc(lcnn_x_edl, lcnn_y_edl), 2)

        resnet_area_ce = round(auc(resnet_x_ce, resnet_y_ce), 2)
        resnet_area_edl = round(auc(resnet_x_edl, resnet_y_edl), 2) 

        lstm_area_ce = round(auc(lstm_x_ce, lstm_y_ce), 2)
        lstm_area_edl = round(auc(lstm_x_edl, lstm_y_edl), 2) 

        plt.plot(lcnn_x_edl, lcnn_y_edl, 'r--') 
        plt.plot(lcnn_x_ce, lcnn_y_ce, 'r')

        plt.plot(resnet_x_edl, resnet_y_edl, 'b--') 
        plt.plot(resnet_x_ce, resnet_y_ce, 'b')

        plt.plot(lstm_x_edl, lstm_y_edl, 'g--') 
        plt.plot(lstm_x_ce, lstm_y_ce, 'g')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        #plt.title('ROC Curve')
        plt.legend(['LCNN EDL AUC: {}'.format(lcnn_area_edl), 
            'LCNN CE AUC: {}'.format(lcnn_area_ce), 'ResNet EDL AUC: {}'.format(resnet_area_edl), 
            'ResNet CE AUC: {}'.format(resnet_area_ce), 'LSTM EDL AUC: {}'.format(lstm_area_edl), 
            'LSTM CE AUC: {}'.format(lstm_area_ce)])

        plt.xlabel('False Positive Ratio')
        plt.ylabel('True Positive Ratio')
        plt.grid(True)
        plt.show()


    def plot_pdfs(self):
        lcnn_x_edl, lcnn_y_edl = self.load_scores(model_typ='lcnn', typ='edl')
        lcnn_x_ce, lcnn_y_ce = self.load_scores(model_typ='lcnn', typ='ce') 

        resnet_x_edl, resnet_y_edl = self.load_scores(model_typ='resnet', typ='edl')
        resnet_x_ce, resnet_y_ce = self.load_scores(model_typ='resnet', typ='ce')

        lstm_x_edl, lstm_y_edl = self.load_scores(model_typ='lstm', typ='edl')
        lstm_x_ce, lstm_y_ce = self.load_scores(model_typ='lstm', typ='ce')
        
        fig, axis = plt.subplots(1, 2, sharey=True, tight_layout=True)
        #n, bins, _ = axis[0].hist([lcnn_x_edl, lcnn_x_ce], histtype='barstacked', bins=25, density=1, stacked=True, color=['r', 'b'], alpha = 0.85)

        y,binEdges=np.histogram(lcnn_x_edl ,bins=25, density=1)
        bins = np.arange(np.floor(y.min()),np.ceil(y.max()))

        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axis[0].plot(bincenters,y,'r-')

        y,binEdges=np.histogram(lcnn_x_ce ,bins=25, density=1)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axis[0].plot(bincenters,y,'b-')
        
        axis[0].legend(['EDL', 'Cross-Entropy'])

        axis[0].set_xlabel('Probability')
        axis[0].set_ylabel('Spoof Scores')
        #axis[0].set_title('Probability Distribution Function of the CE Loss')
        #n, bins, _ = axis[1].hist([lstm_y_edl, lstm_y_ce], histtype='barstacked', bins=25, density=1, stacked=True, color=['r', 'b'], alpha = 0.85)
        
        y,binEdges=np.histogram(lcnn_y_edl ,bins=25, density=1)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axis[1].plot(bincenters,y,'r-')

        y,binEdges=np.histogram(lcnn_y_ce ,bins=25, density=1)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axis[1].plot(bincenters,y,'b-')

        axis[1].legend(['EDL', 'Cross-Entropy'])
        
        axis[1].set_xlabel('Probability')
        axis[1].set_ylabel('Genuine Scores')
        #axis[1].set_title('Probability Distribution Function of the CE Loss')

        axis[0].grid(True)
        axis[1].grid(True)


        #kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=25)

        #plt.hist(lcnn_x_edl, **kwargs)
        #plt.hist(lcnn_x_ce, **kwargs)
    

  

        plt.show()


    def probability_distribution_function_for_all(self, logits, labels, typ=None, write_as_txt=True, model_typ=None):
        L_match, L_not_match = [], []
        for x,y in zip(logits, labels):
            match = self.true_classified_samples_1d(x, y)
            not_match = self.miss_classified_samples_1d(x, y)

            for v in match: L_match.append(v)
            for v in not_match: L_not_match.append(v)

        L_not_match = np.array(L_not_match)
        L_match = np.array(L_match)

        # normalize result 
        #total_match = len(L_match)
        #total_not_match = len(L_not_match)
#
        #y_match = np.arange(0, total_match)
        #y_not_match = np.arange(0,total_not_match)

        #norm_not_match = [num/total_not_match for num in y_not_match]
        #norm_match = [num/total_match for num in y_match]

        if write_as_txt:
            with open('scores_pdf/{}_negatives_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in L_not_match:
                    f.write(str(line) + '\n')

            with open('scores_pdf/{}_positives_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in L_match:
                    f.write(str(line) + '\n')

    

    def probability_distribution_function_class_based(self, logits, labels, typ=None, write_as_txt=True, model_typ=None):
        spoof, genuine = [], []
        for x,y in zip(logits, labels):     
            x, y = self.miss_classified_samples_1d(x, y)       
            for x_, y_ in zip(x,y): 
                if y_ == 0:
                    spoof.append(x_)
                else:
                    genuine.append(x_)

        genuine = np.array(genuine)
        spoof = np.array(spoof)

        if write_as_txt:
            with open('scores_pdf_class_based/{}_spoof_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in spoof:
                    f.write(str(line) + '\n')

            with open('scores_pdf_class_based/{}_genuine_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in genuine:
                    f.write(str(line) + '\n')



    def find_best_eer2(self, epoch=50, plot_roc=None):
        # print all, find best, custom versions
        eer_list = list()
        for e in range(epoch):
            eval_logits, _ , eval_true_labels, _, _  = self.load(typ="eval", e=e)
            eer_list.append(self.compute_eer(eval_logits, eval_true_labels))

        # find minimum eer score 
        min_eer = min(eer_list)
        index_ = eer_list.index(min_eer)
        print('eval eer: {}'.format(min_eer))

        index_ = epoch

        # find AUC value of the eer score
        eval_logits, softmax_, eval_true_labels, sigmoid_, probs_  = self.load(typ="eval", e=index_)
        #self.compute_roc_curve(logits=eval_logits, labels=eval_true_labels, write_as_txt=False, typ='edl', model_typ='lstm')  
        #self.probability_distribution_function_class_based(logits=sigmoid_, labels=eval_true_labels, write_as_txt=True, typ='edl', model_typ='lcnn')
        self.compute_roc_curve_bob(logits=eval_logits, labels=eval_true_labels, write_as_txt=True, typ='edl', model_typ='lcnn')
    

    def cumulative_distribution_function(self, logits, labels, typ, model_typ, write_as_txt=True):
        # from statsmodels.distributions.empirical_distribution import ECDF

        L_match, L_not_match = [], []

        for x,y in zip(logits, labels):
            match = self.true_classified_samples(x, y)
            not_match = self.miss_classified_samples(x, y)
            for v in match: L_match.append(v)
            for v in not_match: L_not_match.append(v)

        L_not_match = np.array(L_not_match)
        L_match = np.array(L_match)

        #miss_classified = self.miss_classified_samples(final_logits, final_labels)
        #true_classified = self.true_classified_samples(final_logits, final_labels)

        List = []
        for logit in L_not_match:
            sum_ = 0
            for elem in logit: # data -> logits vector
                elem = elem.tolist()
                sum_ += -(elem*math.log10(elem))
            List.append(sum_)

        entropy_array = List
        x = np.sort(entropy_array) # sort logits vector 

        # plot logits
        #x_max = [logit.max(axis=1) for logit in logits]
        #x_max = np.array(x_max)    
        #x = x_max[idx]

        y = np.arange(1,len(x)+1)/len(x)

        #ecdf = ECDF(miss_classified)

        #x = ecdf.x
        #y = ecdf.y
        from numpy import trapz
        area = trapz(y, x=x)
        print(area)

        if write_as_txt:  
            with open('scores_cdf/{}_false_x_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in x:
                    f.write(str(line) + '\n')

            with open('scores_cdf/{}_false_y_{}.txt'.format(model_typ, typ), 'a') as f:
                for line in y:
                    f.write(str(line) + '\n')
        else:
            plt.plot(x, y, '--')
            plt.xlabel('Empirical CDF of Entropy in ASV Fail Samples For 50 epoch')
            plt.ylabel('ECDF')
            plt.margins(0.02)
            plt.show()


    def find_best_eer(self, epoch=50):
        eer_list = list()
        for e in range(0, epoch):
            eval_logits, _, eval_true_labels, _  = self.load(typ="eval", e=e)
            eer_list.append(self.compute_eer(eval_logits, eval_true_labels))

        # # find minimum eer score 
        # min_eer = min(eer_list)
        # index_ = eer_list.index(min_eer)
        # print('eval eer: {}'.format(min_eer))
        
        for i in range(len(eer_list)):

            print('========Index: {}'.format(i+1))
            print('eval eer: {}'.format(eer_list[i]))

            # find AUC value of the eer score
            eval_logits, _, eval_true_labels, _  = self.load(typ="eval", e=i)
            self.compute_roc_curve(logits=eval_logits, labels=eval_true_labels)  

    
    def plot_cdfs(self):
        lcnn_edl_x_false, lcnn_edl_y_false = self.load_scores(model_typ='lcnn', typ='edl', tf='false')
        lcnn_edl_x_true, lcnn_edl_y_true = self.load_scores(model_typ='lcnn', typ='edl', tf='true')

        lcnn_ce_x_false, lcnn_ce_y_false = self.load_scores(model_typ='lcnn', typ='ce', tf='false')
        lcnn_ce_x_true, lcnn_ce_y_true = self.load_scores(model_typ='lcnn', typ='ce', tf='true')

        
        fig, axis = plt.subplots(1, 2, sharey=True, tight_layout=True)
        #n, bins, _ = axis[0].hist([lcnn_x_edl, lcnn_x_ce], histtype='barstacked', bins=25, density=1, stacked=True, color=['r', 'b'], alpha = 0.85)

        axis[0].title.set_text('Empirical CDF of EDL Loss')
        axis[0].plot(lcnn_edl_x_false, lcnn_edl_y_false,'k--')
        axis[0].plot(lcnn_edl_x_true, lcnn_edl_y_true,'k')
        axis[0].plot([], [], ' ', label="Extra label on the legend")

        axis[0].legend(['misclassified AUC = 0.08', 'true classified AUC = 0.17', '(true - miss = 0.09)'])

        axis[0].set_xlabel('Entropy')
        axis[0].set_ylabel('Probability')
        #axis[0].set_title('Probability Distribution Function of the CE Loss')
        #n, bins, _ = axis[1].hist([lstm_y_edl, lstm_y_ce], histtype='barstacked', bins=25, density=1, stacked=True, color=['r', 'b'], alpha = 0.85)
        
        axis[1].title.set_text('Empirical CDF of Cross-Entropy Loss')
        axis[1].plot(lcnn_ce_x_false, lcnn_ce_y_false,'k--')
        axis[1].plot(lcnn_ce_x_true, lcnn_ce_y_true,'k')
        axis[1].plot([], [], ' ', label="Extra label on the legend")

        axis[1].legend(['misclassified AUC = 0.12', 'true classified AUC = 0.18', '(true - miss = 0.06)'])

        axis[1].set_xlabel('Entropy')
        axis[1].set_ylabel('Probability')

        #axis[1].set_title('Probability Distribution Function of the CE Loss')

        axis[0].grid(True)
        axis[1].grid(True)

        plt.show()

   

if __name__ == '__main__':
    compute_scores = Evaluation_EER()
    #compute_scores.eer() 
    # compute_scores.cumulative_distribution_function()
    #compute_scores.plot_roc_for_all()
    compute_scores.find_best_eer2(epoch=20)

    #compute_scores.plot_roc_curve()
    #compute_scores.plot_pdfs()
     #compute_scores.plot_cdfs()

    #compute_scores.probability_distribution_function_for_all()
    #compute_scores.compute_roc_curve()
