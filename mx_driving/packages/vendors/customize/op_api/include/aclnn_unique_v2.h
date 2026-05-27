
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_UNIQUE_V2_H_
#define ACLNN_UNIQUE_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnUniqueV2GetWorkspaceSize
 * parameters :
 * input : required
 * outputOut : required
 * uniqueCntOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnUniqueV2GetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *outputOut,
    const aclTensor *uniqueCntOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnUniqueV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnUniqueV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
