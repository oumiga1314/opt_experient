class candicate_key:
    def __init__(self):
         self.kk=0
         self.flag=False

    def candiate(self,k,threshold,flag):
        """
        :param k:给出上次的k
       :param threshold: 候选key的上限
       :param flag: 用来标记是key加还是key减
       :return: 返回调整后的k和标记flag
        """
        flag_temp=flag
        self.kk=0
        if k == 1:

             flag_temp = True
             self.kk=k+1
        elif k <= int(threshold * 0.5):

            if flag:
               self.kk= k + 1
            else:
                self.kk = k - 1
        elif (k<= threshold) & (k > int(threshold * 0.5)):

                if flag:
                    self.kk = k + 1
                else:
                    self.kk = k - 1
        if k>threshold:
                flag_temp = False
                self.kk= int(threshold * 0.5)
        return  self.kk,flag_temp
