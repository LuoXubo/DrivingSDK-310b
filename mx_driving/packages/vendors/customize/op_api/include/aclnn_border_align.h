
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BORDER_ALIGN_H_
#define ACLNN_BORDER_ALIGN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBorderAlignGetWorkspaceSize
 * parameters :
 * input : required
 * rois : required
 * pooledSize : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBorderAlignGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *rois,
    int64_t pooledSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBorderAlign
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBorderAlign(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
