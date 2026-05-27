
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DIFF_IOU_ROTATED_SORT_VERTICES_H_
#define ACLNN_DIFF_IOU_ROTATED_SORT_VERTICES_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDiffIouRotatedSortVerticesGetWorkspaceSize
 * parameters :
 * vertices : required
 * mask : required
 * numValid : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDiffIouRotatedSortVerticesGetWorkspaceSize(
    const aclTensor *vertices,
    const aclTensor *mask,
    const aclTensor *numValid,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDiffIouRotatedSortVertices
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDiffIouRotatedSortVertices(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
