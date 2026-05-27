
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_MEAN_H_
#define ACLNN_SCATTER_MEAN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterMeanGetWorkspaceSize
 * parameters :
 * src : required
 * indices : required
 * var : required
 * dim : required
 * outOut : required
 * countOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterMeanGetWorkspaceSize(
    const aclTensor *src,
    const aclTensor *indices,
    const aclTensor *var,
    int64_t dim,
    const aclTensor *outOut,
    const aclTensor *countOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterMean
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterMean(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
