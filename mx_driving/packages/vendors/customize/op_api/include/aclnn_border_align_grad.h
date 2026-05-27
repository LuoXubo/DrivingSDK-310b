
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BORDER_ALIGN_GRAD_H_
#define ACLNN_BORDER_ALIGN_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBorderAlignGradGetWorkspaceSize
 * parameters :
 * gradOut : required
 * boxes : required
 * argmaxIdx : required
 * channels : required
 * boxSize : required
 * height : required
 * width : required
 * poolSize : required
 * batchSize : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBorderAlignGradGetWorkspaceSize(
    const aclTensor *gradOut,
    const aclTensor *boxes,
    const aclTensor *argmaxIdx,
    int64_t channels,
    int64_t boxSize,
    int64_t height,
    int64_t width,
    int64_t poolSize,
    int64_t batchSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBorderAlignGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBorderAlignGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
