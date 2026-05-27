
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CAL_ANCHORS_HEADING_H_
#define ACLNN_CAL_ANCHORS_HEADING_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCalAnchorsHeadingGetWorkspaceSize
 * parameters :
 * anchors : required
 * originPos : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCalAnchorsHeadingGetWorkspaceSize(
    const aclTensor *anchors,
    const aclTensor *originPos,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCalAnchorsHeading
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCalAnchorsHeading(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
