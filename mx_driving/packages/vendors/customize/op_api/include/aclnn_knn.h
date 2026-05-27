
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_KNN_H_
#define ACLNN_KNN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnKnnGetWorkspaceSize
 * parameters :
 * xyz : required
 * centerXyz : required
 * isFromKnn : required
 * k : required
 * distOut : required
 * idxOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKnnGetWorkspaceSize(
    const aclTensor *xyz,
    const aclTensor *centerXyz,
    bool isFromKnn,
    int64_t k,
    const aclTensor *distOut,
    const aclTensor *idxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnKnn
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKnn(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
