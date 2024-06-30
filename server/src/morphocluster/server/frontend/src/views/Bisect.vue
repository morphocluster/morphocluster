<template>
    <div id="bisect">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark text-light">
            <router-link class="navbar-brand text-light" to="/"
                >MorphoCluster</router-link
            >
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav mr-auto">
                    <li class="nav-item nav-link text-light" v-if="project">
                        {{ project.name }}
                    </li>
                    <li class="nav-item nav-link text-light">Grow</li>
                    <li class="nav-item nav-link text-light" v-if="node">
                        {{ node.name }}
                    </li>
                </ul>
                <dark-mode-control />
            </div>
        </nav>
        <div v-if="node_status == 'loading'">Loading node...</div>
        <div class="bg-light section-heading border-bottom border-top">
            Node members
            <span v-if="node">({{ node.n_objects }} objects)</span>
            <span
                class="float-right mdi mdi-dark mdi-information-outline"
                v-b-tooltip.hover.html
                title="All members of this node, randomly ordered."
            />
        </div>
        <div id="node-members" class="row scrollable">
            <div v-if="node" class="col col-1">
                <member-preview v-bind:member="node" />
            </div>

            <div
                :key="getUniqueId(m)"
                v-for="m of node_members"
                class="col col-1"
            >
                <member-preview v-bind:member="m" />
            </div>

            <infinite-loading
                ref="infload"
                v-if="node"
                @infinite="updateNodeMembers"
                spinner="circles"
            >
                <div slot="no-more">
                    <span v-b-tooltip.hover.html title="End of list."
                        >&#8718;</span
                    >
                </div>
            </infinite-loading>
        </div>
        <div v-if="rec_status == 'loading'">Loading recommendations...</div>
        <div
            v-if="rec_members.length && !done"
            class="bg-light section-heading border-bottom border-top"
        >
            Recommended members
            <span v-if="typeof rec_current_page != 'undefined'"
                >(Page {{ rec_current_page + 1 }} / {{ rec_n_pages }})</span
            >
            <span
                class="float-right mdi mdi-dark mdi-information-outline"
                v-b-tooltip.hover.html
                title="Recommendations for this node, page by page."
            />
        </div>
        <div
            id="recommended-members"
            v-if="rec_members && !done"
            class="row scrollable"
        >
            <div
                class="col col-12 spinner-container"
                v-if="rec_status == 'loading'"
            >
                <spinner spinner="circles" />
            </div>
            <div
                :key="getUniqueId(m)"
                v-for="m of rec_members"
                class="col col-1"
            >
                <member-preview
                    :member="m"
                    :controls="rec_member_controls"
                    v-on:remove="removeMember"
                    v-on:accept="acceptMember"
                />
            </div>
        </div>
        <div v-if="done" class="bg-light section-heading">Report</div>
        <div id="report" v-if="done" class="scrollable">
            Bisection done.
            <table>
                <tr>
                    <th>Total number of pages:</th>
                    <td>{{ rec_n_pages }}</td>
                </tr>
                <tr>
                    <th>Number of valid pages:</th>
                    <td>{{ n_valid_pages }}</td>
                </tr>
                <tr>
                    <th>Number of invalid pages:</th>
                    <td>{{ n_invalid_pages }}</td>
                </tr>
                <tr>
                    <th>Number of rejected members:</th>
                    <td>{{ rejected_members.length }}</td>
                </tr>
            </table>
            <p v-if="n_valid_pages == rec_n_pages">
                You accepted all recommendations. You may want to
                <i>start over</i> to get more.
            </p>
            <p v-if="saving">Your input is being saved...</p>
            <p v-if="saved">
                Your input has been saved. Go on with the next node.
            </p>
            <p v-if="saving_total_ms">
                Saving took {{ saving_total_ms / 1000 }}s.
            </p>
        </div>
        <div id="progress">
            <div :style="{ flexGrow: n_valid_pages }" class="bg-success" />
            <div :style="{ flexGrow: n_unsure_pages }" class="bg-warning" />
            <div :style="{ flexGrow: n_invalid_pages }" class="bg-danger" />
        </div>
        <div
            id="decision"
            v-if="rec_status == 'loaded' && node_status == 'loaded'"
        >
            <b-form-checkbox v-model="turtle_mode">Turtle mode</b-form-checkbox>
            <b-button
                :disabled="saving"
                variant="success"
                v-b-tooltip.hover.html
                title="All visible recommendations match without exception. Increase left limit. <kbd>F</kbd>"
                @click.prevent="membersOk"
            >
                <i class="mdi mdi-check-all" /> OK</b-button
            >
            <b-button
                id="button-not-ok"
                :disabled="saving"
                variant="danger"
                v-b-tooltip.hover.html
                :title="not_ok_tooltip"
                @click.prevent="membersNotOk"
            >
                <i class="mdi mdi-close" /> Not OK</b-button
            >
            <b-button
                :disabled="!saved"
                variant="secondary"
                v-b-tooltip.hover.html
                title="Discard progress and start over. <kbd>R</kbd>"
                @click.prevent="initialize"
            >
                <i class="mdi mdi-restart" /> Start over</b-button
            >
            <!-- <b-button variant="outline-success" v-b-tooltip.hover title="Assign all safe objects to the current node." @click.prevent="saveResult">Save result</b-button> -->
            <!-- <div>
        n_valid_pages: {{n_valid_pages}}, n_unsure_pages: {{n_unsure_pages}}, n_invalid_pages: {{n_invalid_pages}}, rec_interval_left: {{rec_interval_left}}, rec_interval_right: {{rec_interval_right}}
      </div> -->
            <b-button
                :disabled="!saved"
                variant="secondary"
                v-b-tooltip.hover.html
                title="Continue with next node. <kbd>N</kbd>"
                @click.prevent="next"
            >
                <i class="mdi mdi-chevron-right" /> Next
            </b-button>
        </div>
        <message-log class="bg-light" :messages="messages" />
        <b-modal
            ref="doneModal"
            centered
            no-fade
            header-bg-variant="success"
            title="Growing done"
        >
            <div class="d-block text-center">
                Growing is done for this project.
            </div>
            <footer slot="modal-footer">
                <b-button variant="primary" :to="{ name: 'projects' }"
                    >Back to projects</b-button
                >
            </footer>
        </b-modal>
    </div>
</template>

<script>
import axios from "axios";
import shuffle from "lodash/shuffle";

import InfiniteLoading from "vue-infinite-loading";
import Spinner from "vue-infinite-loading/src/components/Spinner";

import mixins from "@/mixins.js";

import * as api from "@/helpers/api.js";

import MemberPreview from "@/components/MemberPreview.vue";
import MessageLog from "@/components/MessageLog.vue";
import DarkModeControl from "@/components/DarkModeControl.vue";

import Vue from "vue";

const MAX_N_RECOMMENDATIONS = 100000;

export default {
    name: "BisectView",
    data() {
        return {
            node_status: "",
            project: null,
            node: null,
            node_members: [],
            node_members_url: null,
            node_members_page: null,
            rec_members: [],
            rejected_members: [],
            /*
            rec_interval_left is the first unsure page.
            */
            rec_interval_left: 0,
            rec_interval_right: null,
            rec_current_page: 0,
            rec_base_url: null,
            rec_n_pages: null,
            rec_request_id: null,
            rec_status: "",
            done: false,
            rec_member_controls: [
                {
                    event: "remove",
                    icon: "mdi-close",
                    title: "Remove this object from the suggestions.",
                },
                {
                    event: "accept",
                    icon: "mdi-check",
                    title: "Accept this object.",
                },
            ],

            /*
            Will be set to true if the right limit of the interval was found,
            i.e. on the first "bad" page.
            */
            found_right: false,
            /*
            Used to update the current page.
            While the right limit of the interval is not found, doubled for every "good" page
            */
            jump_pages: 1,
            saving: false,
            saved: false,
            saving_start_ms: null,
            saving_total_ms: null,
            turtle_mode: false,
            turtle_mode_auto_changed: false,

            /* Accepted members */
            accepted_members_page: [],

            /* Sorting effort */
            log_data: {
                // Number of decisions the user had to make until saving
                // Increased for ok, not ok, accept single, reject single
                n_accept_page: 0,
                n_reject_page: 0,
                n_accept_object: 0,
                n_reject_object: 0,
                // Time when the view is visited
                time_visit: null,
                // Time when the view is fully initialized
                time_initialized: null,
                // Time when the last page is done
                time_done: null,
                // Time when the result was saved
                time_saved: null,
            },
        };
    },
    components: {
        MemberPreview,
        InfiniteLoading,
        MessageLog,
        Spinner,
        DarkModeControl,
    },
    mixins: [mixins],
    watch: {
        $route: "initialize",
        turtle_mode: function (value) {
            if (value) {
                console.log("Turtle mode on.");
                // Reset current page to rec_interval_left
                if (this.rec_current_page != this.rec_interval_left) {
                    this.rec_current_page = this.rec_interval_left;
                    this.showNext();
                }
            } else {
                console.log("Turtle mode off.");
            }
        },
        not_ok_tooltip: function () {
            // Show tooltip when not_ok_tooltip changed
            Vue.nextTick(() => {
                this.$root.$emit("bv::show::tooltip", "button-not-ok");
            });
        }
    },
    created() {
        this.initialize();
    },
    mounted() {
        window.addEventListener("keypress", this.keypress);
    },
    beforeDestroy() {
        window.removeEventListener("keypress", this.keypress);
    },
    computed: {
        n_valid_pages() {
            return this.rec_interval_right - this.n_unsure_pages;
        },
        n_unsure_pages() {
            return Math.max(
                0,
                this.rec_interval_right - this.rec_interval_left
            );
        },
        n_invalid_pages() {
            return this.rec_n_pages - this.rec_interval_right;
        },
        not_ok_tooltip() {
            if (this.accepted_members_page.length) {
                return "<strong>All</strong> visible recommendations <strong>do not match</strong> without exception. Save all as rejected and proceed. <kbd>J</kbd>";
            }
            return "<strong>Some</strong> visible recommendations do not match. Decrease right limit. <kbd>J</kbd>";
        },
    },
    methods: {
        initialize() {
            console.log("Initializing...");

            // Reset data (but keep project)
            Object.assign(this.$data, this.$options.data(), {
                project: this.project,
            });

            // Time when the view is visited
            this.log_data.time_visit = Date.now();

            const project_id = parseInt(this.$route.params.project_id);

            this.node_status = "loading";

            var projectPromise = new Promise((resolve) => {
                if (this.project && this.project.project_id == project_id) {
                    // Project was already loaded.
                    resolve();
                } else {
                    // Load project
                    this.project = null;
                    this.progress = null;
                    api.getProject(project_id, true).then((project) => {
                        this.project = project;
                        this.progress = {
                            n_approved_objects: project.n_approved_objects,
                            n_objects_total: project.n_objects_total,
                        };
                        resolve();
                    });
                }
            });

            var nodeIdPromise = projectPromise.then(() => {
                // If we already have a node_id, return it
                if (this.$route.params.node_id) {
                    return parseInt(this.$route.params.node_id);
                }
                // ... otherwise get the next unfilled node
                return (
                    api
                        .getNextUnfilledNode(this.project.node_id, {
                            leaf: true,
                            preferred_first: true,
                        })
                        // (This really needs to be nested!)
                        .then((node_id) => {
                            if (node_id === null) {
                                // Done
                                this.$refs.doneModal.show();
                                return Promise.reject(
                                    new Error("No next node")
                                );
                            }
                            const to = {
                                name: "bisect",
                                params: {
                                    project_id: project_id,
                                    node_id: node_id,
                                },
                            };

                            // Navigate to the new adress. This starts a new processing of the whole chain.
                            console.log("Navigating to", to);
                            this.$router.replace(to);

                            // Don't process this chain further as there is now a new one
                            throw null;
                        })
                );
            });

            nodeIdPromise
                .then((node_id) => {
                    return api.getNode(node_id).then((node) => {
                        this.node = node;
                    });
                })
                .then(() => {
                    this.node_status = "loaded";
                })
                .catch((e) => {
                    this.axiosErrorHandler(e);
                });

            nodeIdPromise
                .then((node_id) => {
                    console.log("getNodeRecommendedObjects...");
                    this.rec_status = "loading";

                    return api.getNodeRecommendedObjects(node_id, {
                        max_n: MAX_N_RECOMMENDATIONS,
                    });
                })
                .then((data) => {
                    // TODO: Do something when there are no recommendations!
                    this.rec_members = shuffle(data.data);
                    this.rec_base_url = data.links.self;
                    this.rec_n_pages = this.rec_interval_right =
                        data.meta.last_page + 1;

                    this.rec_current_page = this.rec_interval_left = 0;
                    this.rec_status = "loaded";
                    this.rec_request_id = data.meta.request_id;

                    // Time when the view is fully initialized
                    this.log_data.time_initialized = Date.now();
                })
                .catch((e) => {
                    this.axiosErrorHandler(e);
                });
        },
        // updateNodeMembers gets called as an infinite loading handler.
        updateNodeMembers($state) {
            if (!this.node) {
                $state.reset();
                return;
            }
            console.log("updateNodeMembers");

            // Should node_members_url be updated (with unique id etc.) on response?
            var updateMembersUrl = false;

            // TODO: arrange_by=random
            if (!this.node_members_url) {
                const nodes = !!this.node.children;
                this.node_members_url = `/api/nodes/${
                    this.node.node_id
                }/members?objects=${!nodes}&nodes=${nodes}&arrange_by=random&`;
                this.node_members_page = 0;
                updateMembersUrl = true;
            }

            axios
                .get(`${this.node_members_url}&page=${this.node_members_page}`)
                .then((response) => {
                    this.node_members = this.node_members.concat(
                        response.data.data
                    );

                    if (updateMembersUrl) {
                        this.node_members_url = response.data.links.self;
                    }

                    $state.loaded();

                    if (this.node_members_page < response.data.meta.last_page) {
                        this.node_members_page += 1;
                    } else {
                        $state.complete();
                    }
                })
                .catch((e) => {
                    this.axiosErrorHandler(e);
                });
        },
        membersOk: function () {
            // Increase number of decisions
            this.log_data.n_accept_page++;

            this.rec_interval_left = this.rec_current_page + 1;

            this.updateCurrentPage();

            this.showNext();
        },
        membersNotOk: function () {
            // Increase umber of decisions
            this.log_data.n_reject_page++;

            if (this.accepted_members_page.length) {
                // If there are accepted members, reject all remaining and proceed like in membersOk
                var remaining_members = this.rec_members.map(this.getUniqueId);

                console.log("Rejecting", remaining_members);
                this.rejected_members.push(...remaining_members);

                this.rec_interval_left = this.rec_current_page + 1;
            } else {
                this.rec_interval_right = this.rec_current_page;
                this.found_right = true;
            }

            // Update page, but go to first quarter instead of half of the interval.
            this.updateCurrentPage(0.25);

            this.showNext();
        },
        updateCurrentPage(frac = 0.5) {
            if (this.turtle_mode) {
                // In turtle mode, only go one page forward.
                this.rec_current_page = Math.min(
                    this.rec_interval_left,
                    this.rec_n_pages - 1
                );
            } else if (!this.found_right) {
                // If the right side of the interval was not found yet, jump forward
                // and increase leap.
                this.rec_current_page = Math.min(
                    this.rec_current_page + this.jump_pages,
                    this.rec_n_pages - 1
                );
                this.jump_pages *= 2;
            } else {
                // Otherwise perform regular bisection
                this.rec_current_page = Math.trunc(
                    (1 - frac) * this.rec_interval_left +
                        frac * this.rec_interval_right
                );
            }
        },
        showNext: function () {
            console.log(
                this.rec_current_page,
                this.rec_interval_left,
                this.rec_interval_right
            );

            if (this.n_unsure_pages <= 0) {
                this.done = true;

                this.saveResult();
                return;
            }

            this.accepted_members_page = [];

            axios
                .get(`${this.rec_base_url}&page=${this.rec_current_page}`)
                .then((response) => {
                    console.log(
                        response.data.data,
                        shuffle(response.data.data)
                    );
                    this.rec_members = shuffle(response.data.data);
                })
                .catch((e) => {
                    this.axiosErrorHandler(e);
                });
        },
        saveResult() {
            // For each page in the valid range (0<=x<this.rec_interval_left):
            // Fetch members and assign to the current node.

            if (this.saving) {
                console.log("Saving already in progress!");
                return;
            }

            console.log("Saving...");
            this.saving = true;

            // Time when the last page is done
            this.log_data.time_done = Date.now();

            // Save all data of the current run.
            // If the user continues with the next node, all data is lost.
            var node = this.node;

            api.nodeAcceptRecommendations(
                node.node_id,
                this.rec_request_id,
                this.rejected_members,
                this.rec_interval_left - 1
            )
                .then(() => {
                    console.log("Saved all recommendations.");
                })
                .then(() => {
                    return api.patchNode(node.node_id, { filled: true });
                })
                .then(() => {
                    // Update progress
                    api.getNodeProgress(node.node_id, {
                        log: "grow",
                    });
                })
                .then(() => {
                    console.log("Saved.");
                    this.saving = false;
                    this.saved = true;
                    // Time when the result was saved
                    this.log_data.time_saved = Date.now();
                    this.saving_total_ms =
                        this.log_data.time_saved - this.log_data.time_done;
                    this.messages.unshift(`Saved ${node.node_id}.`);

                    // Finally log everything (including save timings)
                    return api.log(
                        "grow_saved",
                        node.node_id,
                        null,
                        this.log_data
                    );
                })
                .catch((e) => {
                    this.messages.unshift(`Error saving ${node.node_id}.`);
                    console.log(e);
                });
        },
        hideMember(member) {
            var index = this.rec_members.indexOf(member);
            if (index > -1) {
                this.rec_members.splice(index, 1);
            }
        },
        autoEnableTurtleMode() {
            if (!this.turtle_mode_auto_changed) {
                this.turtle_mode = true;
                this.turtle_mode_auto_changed = true;
            }
        },
        removeMember(member) {
            console.log("Reject", this.getUniqueId(member));

            // Increase number of decisions
            this.log_data.n_reject_object++;

            // Remove from current recommendations
            this.hideMember(member);

            // Enable turtle mode
            this.autoEnableTurtleMode();

            // And add to rejected
            this.rejected_members.push(this.getUniqueId(member));
        },
        acceptMember(member) {
            console.log("Accept", this.getUniqueId(member));

            // Increase umber of decisions
            this.log_data.n_accept_object++;

            // Remove from current recommendations
            this.hideMember(member);

            // Don't enable turtle mode here as we might accept all
            // Enable turtle mode
            this.autoEnableTurtleMode();

            this.accepted_members_page.push(this.getUniqueId(member));
            this.messages.unshift(
                `Accepted ${this.accepted_members_page.length} objects.`
            );
        },
        next() {
            this.$router.push({
                name: "bisect",
                params: { project_id: this.project.project_id },
            });
        },
        keypress(event) {
            if (
                this.node_status != "loaded" ||
                this.rec_status != "loaded" ||
                event.altKey ||
                event.ctrlKey ||
                event.metaKey ||
                event.shiftKey
            ) {
                return;
            }
            if (event.key == "f") {
                this.membersOk();
            } else if (event.key == "j") {
                this.membersNotOk();
            } else if (event.key == "r") {
                this.initialize();
            } else if (event.key == "n" && this.done) {
                console.log("next");
                this.next();
            }
        },
    },
};
</script>

<style>
#bisect {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#bisect > * {
    padding: 0 10px;
}

.scrollable {
    margin: 0;
    overflow-y: auto;
}

#decision {
    margin: 0 auto;
}

#decision button {
    margin: 0 1em;
}

/* #messages {
  overflow-y: auto;
  height: 3em;
} */

#node-members .col,
#recommended-members .col {
    padding: 0 5px;
}

#node-members {
    flex: 1;
}

#recommended-members,
#report {
    flex: 2;
}

#progress {
    display: flex;
    flex-wrap: nowrap;
    margin: 0.2em 0;
}

#progress div {
    height: 0.2em;
}

.section-heading {
    margin: 0.2em 0;
}

.spinner-container {
    text-align: center;
    margin: 28px 0;
}
</style>
