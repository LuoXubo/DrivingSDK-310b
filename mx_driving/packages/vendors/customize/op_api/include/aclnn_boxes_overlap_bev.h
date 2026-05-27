
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BOXES_OVERLAP_BEV_H_
#define ACLNN_BOXES_OVERLAP_BEV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBoxesOverlapBevGetWorkspaceSize
 * parameters :
 * boxesA : required
 * boxesB : required
 * formatFlag : optional
 * clockwise : optional
 * modeFlag : optional
 * aligned : optional
 * margin : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBoxesOverlapBevGetWorkspaceSize(
    const aclTensor *boxesA,
    const aclTensor *boxesB,
    int64_t formatFlag,
    bool clockwise,
    int64_t modeFlag,
    bool aligned,
    double margin,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBoxesOverlapBev
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBoxesOverlapBev(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
