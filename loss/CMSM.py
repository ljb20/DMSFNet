import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Compute_cmpm(nn.Module):
    def __init__(self):
        super(Compute_cmpm,self).__init__()
    def forward(image_embeddings, IR_embeddings, labels, epsilon=1e-8):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype tombeddings: Tensor with dtype torch.float32
        :param labelsrch.float32
        :param text_e: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        # labels1_reshape = torch.reshape(label1, (batch_size, 1))
        # labels2_reshape = torch.reshape(label2, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        # labels2_dist = labels2_reshape - labels2_reshape.t()
        labels1_mask = (labels_dist == 0).float()
        # labels2_mask = (labels2_dist == 0).float()
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = IR_embeddings / IR_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(IR_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels1_mask_norm = labels1_mask / labels1_mask.norm(dim=1)
        # labels2_mask_norm = labels2_mask / labels2_mask.norm(dim=1)
        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels1_mask_norm + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels1_mask_norm + epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return cmpm_loss

class Loss_CMSM(nn.Module):
    def __init__(self):
        super(Loss_CMSM,self).__init__()
    def forward(self,image_fetures, text_fetures, labels,logit_scale,image_id=None, factor=0.3, epsilon=1e-8):
        """
        Similarity Distribution Matching
        """
        batch_size = image_fetures.shape[0]
            # pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
            # pid_dist = pid - pid.t()
            # labels = (pid_dist == 0).float()
        label1,label2=labels.chunk(2,0)
            # labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels1_reshape = torch.reshape(label1, (batch_size, 1))
        labels2_reshape = torch.reshape(label2, (batch_size, 1))
        labels1_dist = labels1_reshape - labels1_reshape.t()
        labels2_dist = labels2_reshape - labels2_reshape.t()
        labels1_mask = (labels1_dist == 0).float()
        labels2_mask = (labels2_dist == 0).float()
        if image_id != None:
                # print("Mix PID and ImageID to create soft label.")
            image_id = image_id.reshape((-1, 1))
            image_id_dist = image_id - image_id.t()
            image_id_mask = (image_id_dist == 0).float()
            labels = (labels - image_id_mask) * factor + image_id_mask
                # labels = (labels + image_id_mask) / 2

        image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        image_proj_text = logit_scale * i2t_cosine_theta

            # normalize the true matching distribution
            # labels_distribute = labels / labels.sum(dim=1)
        labels1_mask_norm = labels1_mask / labels1_mask.norm(dim=1)
        labels2_mask_norm = labels2_mask / labels2_mask.norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels1_mask_norm + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels2_mask_norm + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss