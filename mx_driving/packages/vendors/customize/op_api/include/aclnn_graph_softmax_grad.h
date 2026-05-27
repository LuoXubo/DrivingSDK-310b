
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GRAPH_SOFTMAX_GRAD_H_
#define ACLNN_GRAPH_SOFTMAX_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGraphSoftmaxGradGetWorkspaceSize
 * parameters :
 * index : required
 * softmaxOutput : required
 * gradOutput : required
 * reduceSum : required
 * nodeNum : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGraphSoftmaxGradGetWorkspaceSize(
    const aclTensor *index,
    const aclTensor *softmaxOutput,
    const aclTensor *gradOutput,
    const aclTensor *reduceSum,
    int64_t nodeNum,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGraphSoftmaxGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGraphSoftmaxGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
