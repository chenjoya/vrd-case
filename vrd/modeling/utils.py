import torch
from torchvision.ops import box_iou

def split(ds, lengths):
    segments = []
    begin = 0
    for i in lengths:
        end = begin + i
        segments.append(ds[begin:end])
        begin = end
    return segments

def pairwise_pbboxes(bboxes, bscores, top=None, bcats=None):
    # bboxes: tensor, N x 4
    matrix = torch.matmul(
        bscores.view(-1, 1),
        bscores.view(1, -1),
    )
    
    # do not calc union box with self
    mask = torch.eye(len(bboxes), 
        device=bboxes.device, dtype=torch.bool)
    matrix[mask] = 0
    
    sidxs, oidxs = matrix.nonzero(as_tuple=True)
    soscores = matrix[sidxs, oidxs]
    soscores, ranks = soscores.sort(descending=True)
    if top is not None:
        ranks = ranks[:top]
        soscores = soscores[:top]
    sidxs, oidxs = sidxs[ranks], oidxs[ranks]
    
    sbboxes = bboxes[sidxs]
    obboxes = bboxes[oidxs]
    if bcats is not None:
        scats = bcats[sidxs]
        ocats = bcats[oidxs]
    # union bboxes
    pbboxes = torch.stack(
        [
            torch.min(sbboxes[:, 0], obboxes[:, 0]),
            torch.min(sbboxes[:, 1], obboxes[:, 1]),
            torch.max(sbboxes[:, 2], obboxes[:, 2]),
            torch.max(sbboxes[:, 3], obboxes[:, 3])
        ],
        dim=1
    )
    if bcats is None:
        return sbboxes, pbboxes, obboxes
    else:
        return sbboxes, scats, pbboxes, obboxes, ocats, soscores

def fake_loss(network):
    # we produce a fake loss
    fake_images = torch.zeros(1, 3, 224, 224).cuda()
    fake_loss = network(fake_images).sum() * 0
    return fake_loss
    
def triplet_matmul(a, b, c):
    batch = a.shape[0]
    ab = torch.matmul(a.unsqueeze(2), b.unsqueeze(1))
    abc = torch.matmul(ab.view(batch, -1).unsqueeze(2), c.unsqueeze(1))
    return abc.view(batch, a.shape[1], b.shape[1], c.shape[1])

def unique_pass(network, transform, image, bboxes):
    bboxes, inverses = bboxes.unique(sorted=False, 
        return_inverse=True, dim=0)

    images = torch.stack(
        [transform(image[:, ymin:ymax, xmin:xmax]) 
        for xmin, ymin, xmax, ymax in bboxes]
    )
    
    features = network(images)
    
    return features[inverses]

def match_and_sample(sbboxes, scats, pbboxes, obboxes, ocats, 
    gt_rs, num_samples, neg2pos, num_classes):
    # match strategy: 
    # (1) we match the union bboxes > 0.5, with the correct category
    # optional:
    # (2) we match if subject and object iou > 0.5
    pcats = -torch.ones(
        len(pbboxes),
        device=pbboxes.device, 
        dtype=torch.long
    )

    pious = box_iou(pbboxes, gt_rs.pbboxes)

    iss, jss = (pious >= 0.5).nonzero(as_tuple=True)
    mask = (scats[iss] == gt_rs.scats[jss]) & (ocats[iss] == gt_rs.ocats[jss])
    pcats[iss[mask]] = gt_rs.pcats[jss[mask]]
    
    pbboxes = torch.cat([pbboxes, gt_rs.pbboxes])
    pcats = torch.cat([pcats, gt_rs.pcats])

    pbboxes, inverses = pbboxes.unique(
        sorted=False, return_inverse=True, dim=0
    )  
    plabels = torch.zeros(len(pbboxes), num_classes,
        device=pbboxes.device, dtype=torch.float)
    for inv, pcat in zip(inverses, pcats):
        if pcat >= 0:
            plabels[inv, pcat] += 1
    
    plabels = (plabels > 0).float()
    # sample
    mask = plabels.sum(dim=1)
    pos_mask = mask > 0
    neg_mask = ~pos_mask
    pos_idxs = pos_mask.nonzero(as_tuple=True)[0]
    neg_idxs = neg_mask.nonzero(as_tuple=True)[0]
    
    pos_numel, neg_numel = pos_idxs.numel(), neg_idxs.numel()

    # rand perm
    neg_idxs = neg_idxs[torch.randperm(neg_numel, device=neg_idxs.device)]
    num_neg = min(num_samples - pos_numel, pos_numel * neg2pos)
    neg_idxs = neg_idxs[:num_neg]

    idxs = torch.cat([pos_idxs, neg_idxs])
    pbboxes = pbboxes[idxs]
    plabels = plabels[idxs]
    
    return pbboxes, plabels

def match(sbboxes, scats, pbboxes, obboxes, ocats, soscores,
    gt_rs, num_samples, neg2pos, num_classes):
    # match strategy: 
    # (1) we match the union bboxes > 0.5, with the correct category
    # optional:
    # (2) we match if subject and object iou > 0.5
    pcats = -torch.ones(
        len(pbboxes),
        device=pbboxes.device, 
        dtype=torch.long
    )

    sious = box_iou(sbboxes, gt_rs.sbboxes)
    pious = box_iou(pbboxes, gt_rs.pbboxes)
    oious = box_iou(obboxes, gt_rs.obboxes)

    # union box >= 0.5
    iss, jss = (pious >= 0.5).nonzero(as_tuple=True)
    # have correct subject and object category, iou >=0.5
    mask = (scats[iss] == gt_rs.scats[jss]) & (ocats[iss] == gt_rs.ocats[jss]) & \
           (sious[iss, jss] >= 0.5) & (oious[iss, jss] >= 0.5)
    pcats[iss[mask]] = gt_rs.pcats[jss[mask]]
    
    # add gt
    sbboxes = torch.cat([sbboxes, gt_rs.sbboxes])
    scats = torch.cat([scats, gt_rs.scats])
    pbboxes = torch.cat([pbboxes, gt_rs.pbboxes])
    pcats = torch.cat([pcats, gt_rs.pcats])
    obboxes = torch.cat([obboxes, gt_rs.obboxes])
    ocats = torch.cat([ocats, gt_rs.ocats])
    soscores = torch.cat(
        [soscores, torch.ones(len(gt_rs.scats), device=soscores.device)]
    )

    mask = pcats >= 0

    pos_idxs = mask.nonzero(as_tuple=True)[0]
    neg_idxs = (~mask).nonzero(as_tuple=True)[0]

    pos_numel, neg_numel = pos_idxs.numel(), neg_idxs.numel()
    pos_idxs = pos_idxs[torch.randperm(pos_numel, device=pos_idxs.device)]
    neg_idxs = neg_idxs[torch.randperm(neg_numel, device=neg_idxs.device)]

    pos_idxs = pos_idxs[:num_samples]
    pos_numel = pos_idxs.numel()
    num_neg = min(num_samples - pos_numel, pos_numel * neg2pos)
    neg_idxs = neg_idxs[:num_neg]
    
    # we have other?
    idxs = torch.cat([pos_idxs, neg_idxs])
    
    return sbboxes[idxs], scats[idxs], pbboxes[idxs], pcats[idxs], obboxes[idxs], ocats[idxs], soscores[idxs]