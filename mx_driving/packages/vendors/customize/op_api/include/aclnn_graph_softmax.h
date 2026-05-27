
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GRAPH_SOFTMAX_H_
#define ACLNN_GRAPH_SOFTMAX_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGraphSoftmaxGetWorkspaceSize
 * parameters :
 * src : required
 * index : required
 * n : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGraphSoftmaxGetWorkspaceSize(
    const aclTensor *src,
    const aclTensor *index,
    int64_t n,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGraphSoftmax
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGraphSoftmax(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
