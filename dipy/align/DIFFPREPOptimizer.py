# Authors: Okan, Anh Thai
import numpy as np
_NQUADPARAMS = 21

class  DIFFPREPOptimizer(object):
    def __init__(self, similarity_metric, x0, opt_flags, grad_params = None):
        self.metric = similarity_metric
        self.Epsilon=0.00001
        self.CurrentIteration = 0
        self.OptimizationFlag = opt_flags


        self.Gradient = np.zeros(_NQUADPARAMS)
        self.BracketParams = np.zeros(7)
        self.BracketParams[0]=0.1
        self.BracketParams[1]=0.0001
        self.BracketParams[2]=0.0001
        self.BracketParams[3]=0.000001
        self.BracketParams[4]=0.000001
        self.BracketParams[5]=0.000000001
        self.BracketParams[6]=0.000000001

        if grad_params is None:
            self.grad_params= np.zeros(_NQUADPARAMS)
            self.grad_params[0]=2.5
            self.grad_params[1]=2.5
            self.grad_params[2]=2.5
            self.grad_params[3]=0.04
            self.grad_params[4]=0.04
            self.grad_params[5]=0.04
            self.grad_params[6]=0.02
            self.grad_params[7]=0.02
            self.grad_params[8]=0.02
            self.grad_params[9]=0.002
            self.grad_params[10]=0.002
            self.grad_params[11]=0.002
            self.grad_params[12]=0.0007
            self.grad_params[13]=0.0007
            self.grad_params[14]=0.0001
            self.grad_params[15]=0.00002
            self.grad_params[16]=0.00002
            self.grad_params[17]=0.00002
            self.grad_params[18]=0.00002
            self.grad_params[19]=0.00002
            self.grad_params[20]=0.00002
        else:
            self.grad_params=grad_params


        self.mode_ids=[]
        self.mode_ids.append([0,1,2])
        self.mode_ids.append([3,4,5])
        self.mode_ids.append([6,7,8])
        self.mode_ids.append([9,10,11])
        self.mode_ids.append([12,13])
        self.mode_ids.append([14])
        self.mode_ids.append([15,16,17,18,19,20])

        self.orig_grad_params=self.grad_params
        self.m_NumberOfIterations=50
        self.m_NumberHalves=5
        self.xopt = x0              # image_metriv-> GetParameters()

        self.Value = 0              # m_Value
        self.CurrentCost= self.metric.distance(self.xopt)       #m_CurrentCost

        self.ResumeOptimization()



    def ResumeOptimization(self):
        m_Stop = False
        spaceDimension = self.Gradient.shape[0]

        last_Cost = self.CurrentCost
        curr_halve = 0

        while (not m_Stop):
            try:
                for mode in range(7):
                    nrm = self.GetGrad(self.mode_ids[mode])
                    if nrm >0:
                        x_f_pair = self.BracketGrad(self.BracketParams[mode])
                        if np.abs(x_f_pair[0,0] - x_f_pair[1,0]) > 0:
                            step_length, new_cost = self.GoldenSearch(self.BracketParams[mode], x_f_pair)
                            self.xopt = self.xopt - step_length * self.Gradient

                            self.Value = new_cost
                            self.CurrentCost = new_cost

                        else:
                            origin_params = self.xopt.copy()
                            temp_change = self.Gradient*self.grad_params*0.01

                            for i in range(5):
                                temp_trans = origin_params - temp_change

                                temp_cost = self.metric.distance(temp_trans)
                                if temp_cost < self.CurrentCost:
                                    self.CurrentCost = temp_cost
                                    self.Value = temp_cost
                                    self.xopt = temp_trans.copy()
                                    break
                                else:
                                    temp_change = temp_change /2


                if (last_Cost - self.CurrentCost) > self.Epsilon:
                    self.CurrentIteration += 1
                    last_Cost = self.CurrentCost
                else:
                    if curr_halve < self.m_NumberHalves:
                        curr_halve +=1
                        self.grad_params /= 1.7
                    else:
                        m_Stop = True
                        break
                if self.CurrentIteration > self.m_NumberOfIterations:
                    m_Stop = True
                    break
            except:
                print("ERROR at iteration ",self.CurrentIteration)


    def GetGrad(self, ids):
        for v in range(len(ids)):

            curr_param_id = ids[v]
            if self.OptimizationFlag[curr_param_id]:
                temp_params = self.xopt.copy()

                temp_params[curr_param_id] += self.grad_params[curr_param_id]
                fp = self.metric.distance(temp_params)

                temp_params[curr_param_id] -= 2 * self.grad_params[curr_param_id]
                fm = self.metric.distance(temp_params)

                self.Gradient[curr_param_id] = (fp - fm) / (2 * self.grad_params[curr_param_id])

        nrm = np.linalg.norm(self.Gradient)

        if nrm > 0:
            for v in range(len(ids)):
                index = ids[v]
                self.Gradient[index] = self.Gradient[index] / nrm

        return nrm

    def BracketGrad(self, brk_const):
        x_f_pairs = np.zeros((3,2))
        MAX_ITERATION = 50
        m_ERR_MARG = 0.00001
        f_ini = self.CurrentCost
        x_ini = 0
        f_min = f_ini
        x_min = x_ini

        x_last = 0
        f_last = 0
        bail = 0
        counter = 1

        while (not bail):
            konst = counter * counter * brk_const
            temp_params = self.xopt - konst*self.Gradient
            f_last = self.metric.distance(temp_params)
            x_last = konst

            if f_last < f_min:
                f_min = f_last
                x_min = x_last

            else:
                if (f_last > f_min + m_ERR_MARG) or (counter > MAX_ITERATION):
                    bail = True

            counter +=1
        x_f_pairs[0,:] = [x_ini, f_ini]
        x_f_pairs[1,:] = [x_min, f_min]
        x_f_pairs[2,:] = [x_last, f_last]

        return x_f_pairs

    def GoldenSearch(self,cst, x_f_pairs):
        MAX_IT = 100
        counter = 0
        TOL = cst
        R = 0.61803399
        C = 1 - R

        ax, bx, cx  = x_f_pairs[:,0]

        x0 = ax
        x3 = cx

        if (np.abs(cx -bx) > np.abs(bx - ax)):
            x1 = bx
            x2 = bx + C*(cx - bx)
        else:
            x2 = bx
            x1 = bx - C*(bx - ax)

        temp_trans = self.xopt - x1 * self.Gradient
        fx1 = self.metric.distance(temp_trans)

        temp_trans = self.xopt - x2 * self.Gradient
        fx2 = self.metric.distance(temp_trans)

        while ((np.abs(x3 - x0)) > TOL) and (counter < MAX_IT):
            if fx2 < fx1:
                x0 = x1
                x1 = x2
                x2 = R * x2 + C * x3
                temp_trans = self.xopt - x2 * self.Gradient
                fxt = self.metric.distance (temp_trans)
                fx1 = fx2
                fx2 = fxt
            else:
                x3 = x2
                x2 = x1
                x1 = R * x1 + C * x0
                temp_trans = self.xopt.copy() - x1 * self.Gradient
                fxt = self.metric.distance (temp_trans)
                fx2 = fx1
                fx1 = fxt

            counter +=1
        new_cost = min(fx1, fx2)
        if fx1 < fx2:
            return x1, new_cost
        else:
            return x2, new_cost