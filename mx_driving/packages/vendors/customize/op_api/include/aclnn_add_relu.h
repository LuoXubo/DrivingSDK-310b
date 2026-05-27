
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_RELU_H_
#define ACLNN_ADD_RELU_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddReluGetWorkspaceSize
 * parameters :
 * xRef : required
 * y : required
 * xRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddReluGetWorkspaceSize(
    aclTensor *xRef,
    const aclTensor *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddRelu
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddRelu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
