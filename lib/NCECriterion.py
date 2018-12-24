import torch
from torch import nn

eps = 1e-7


class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets, Z):
        batchSize = x.size(0)
        K = x.size(1) - 1
        K1 = 4096
        # x.size() 包括了positive和negative samples的数目。
        # K为 negative sample 的数目。
        Pnt = 1 / float(self.nLem)
        # Pnt 为 Pn(i) = 1/n
        Pns = 1 / float(self.nLem)
        lnPmtsum = torch.cuda.FloatTensor(1).zero_()
        lnPonsum = torch.cuda.FloatTensor(1).zero_()
        min_prob_of_group = 0.001

        for i in range(batchSize):

            for j in range(1,5):
                if x[i][j] < min_prob_of_group:
                    break

            #j =1
            for index in range(j):
                #if index > int(10):
                    #break
                # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
                if index == 0:
                    Pmt = x[i].select(0, index)
                    Pmt_div = Pmt.add(K1 * Pnt + eps)
                    lnPmt = torch.div(Pmt, Pmt_div).log_()
                else:

                    Pmt = x[i].select(0, index)
                    Z_temp = Z - x[i].select(0, 0).item() * Z + Pmt.item()
                    Pmt = Pmt * (Z / Z_temp)
                    Pmt_div = Pmt.add(K1 * Pnt + eps)
                    lnPmt_temp = torch.div(Pmt, Pmt_div).log_()
                    lnPmt = torch.add(lnPmt, lnPmt_temp)
                # Pmt 即公式里的 P(i|v)

                if i == 0:
                    lnPmtsum = lnPmt
                else:
                    lnPmtsum = torch.add(lnPmtsum,lnPmt)
                # lnPmt = h(i|v) = P(i|v) / P(i|v) + m*Pn(i)
            '''
                if index == 0:
                    lnPmt_total = lnPmt.log_()
                #lnPon_total = lnPon.log_()
                else:
                    lnPmt_total.add(lnPmt.log_())
                #lnPon_total.add(lnPon.log_())
            '''
            # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
            Pon_div = x[i].narrow(0, j, int(K-j)).add(K1 * Pns + eps)
            Pon = Pon_div.clone().fill_(K1 * Pns)
            lnPon = torch.div(Pon, Pon_div)
            lnPon.log_()

            lnPon = lnPon.sum(0)
            if i == 0:
                lnPonsum = lnPon
            else:
                lnPonsum = torch.add(lnPonsum,lnPon)
        # h(i|v')

        # equation 6 in ref. A


        #lnPmtsum = lnPmt.sum(0)
        #lnPonsum = lnPon.view(-1, 1).sum(0)

        loss = - (lnPmtsum + lnPonsum )/ batchSize
        # loss = -E{log(h(i|v))} - E{log(1 - h(i|v'))}
        return loss

