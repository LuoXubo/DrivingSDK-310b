
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEFORMABLE_CONV2D_V2_H_
#define ACLNN_DEFORMABLE_CONV2D_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDeformableConv2dV2GetWorkspaceSize
 * parameters :
 * x : required
 * offset : required
 * maskOptional : optional
 * weight : required
 * biasOptional : optional
 * kernelSize : required
 * stride : required
 * padding : required
 * dilation : required
 * groups : required
 * deformableGroups : required
 * modulated : required
 * withBias : required
 * yOut : required
 * offsetOutputOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableConv2dV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *offset,
    const aclTensor *maskOptional,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    const aclIntArray *kernelSize,
    const aclIntArray *stride,
    const aclIntArray *padding,
    const aclIntArray *dilation,
    int64_t groups,
    int64_t deformableGroups,
    bool modulated,
    bool withBias,
    const aclTensor *yOut,
    const aclTensor *offsetOutputOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDeformableConv2dV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableConv2dV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
