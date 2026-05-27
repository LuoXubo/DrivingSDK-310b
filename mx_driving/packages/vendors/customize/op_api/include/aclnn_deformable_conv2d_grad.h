
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEFORMABLE_CONV2D_GRAD_H_
#define ACLNN_DEFORMABLE_CONV2D_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDeformableConv2dGradGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * biasOptional : optional
 * offset : required
 * maskOptional : optional
 * offsetOutput : required
 * gradY : required
 * kernelSize : required
 * stride : required
 * padding : required
 * dilation : required
 * groups : required
 * deformableGroups : required
 * modulated : required
 * withBias : required
 * gradXOut : required
 * gradWeightOut : required
 * gradBiasOutOptional : optional
 * gradOffsetOut : required
 * gradMaskOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableConv2dGradGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    const aclTensor *offset,
    const aclTensor *maskOptional,
    const aclTensor *offsetOutput,
    const aclTensor *gradY,
    const aclIntArray *kernelSize,
    const aclIntArray *stride,
    const aclIntArray *padding,
    const aclIntArray *dilation,
    int64_t groups,
    int64_t deformableGroups,
    bool modulated,
    bool withBias,
    const aclTensor *gradXOut,
    const aclTensor *gradWeightOut,
    const aclTensor *gradBiasOutOptional,
    const aclTensor *gradOffsetOut,
    const aclTensor *gradMaskOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDeformableConv2dGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableConv2dGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
