<template>
    <div id="grow">
        <div class="grey lighten-2 section-heading elevation-1">
            Node members
            <span v-if="node">({{ node.n_objects_own }} objects)</span>
            <v-tooltip bottom>
                <template v-slot:activator="{ on }">
                    <div v-on="on" class="float-right mdi mdi-dark mdi-information-outline" />
                </template>
                <span>All members of this node, randomly ordered.</span>
            </v-tooltip>
        </div>
        <div id="node-members" class="row scrollable">
            <!--<div v-if="node" class="col col-1">
                <member-preview :member="node" />
            </div>-->

            <div :key="getUniqueId(m)" v-for="m of node_members" class="col col-1">
                <member-preview :member="m" />
            </div>

            <infinite-loading
                ref="infload"
                v-if="node"
                @infinite="updateNodeMembers"
                spinner="circles"
            >
                <div slot="no-more">
                    <v-tooltip top>
                        <template v-slot:activator="{ on }">
                            <span v-on="on">&#8718;</span>
                        </template>
                        <span>End of list.</span>
                    </v-tooltip>
                </div>
            </infinite-loading>
        </div>
        <div
            v-if="rec_members !== null && !done"
            class="grey lighten-2 section-heading elevation-1"
        >
            Recommended members
            <span
                v-if="typeof rec_current_page != 'undefined'"
            >(Page {{ rec_current_page + 1 }} / {{ rec_n_pages }})</span>
            <v-tooltip>
                <template v-slot:activator="{ on }">
                    <span v-on="on" class="float-right mdi mdi-dark mdi-information-outline" />
                </template>
                <span>Recommendations for this node, page by page.</span>
            </v-tooltip>
        </div>
        <div id="recommended-members" v-if="rec_members !== null && !done" class="row scrollable">
            <div :key="getUniqueId(m)" v-for="m of rec_members" class="col col-1">
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
            <p v-if="saved">Your input has been saved. Go on with the next node.</p>
            <p v-if="saving_total_ms">Saving took {{ saving_total_ms / 1000 }}s.</p>
        </div>
        <div id="progress">
            <div :style="{ flexGrow: n_valid_pages }" class="bg-success" />
            <div :style="{ flexGrow: n_unsure_pages }" class="bg-warning" />
            <div :style="{ flexGrow: n_invalid_pages }" class="bg-danger" />
        </div>
        <div id="decision" v-if="rec_status == 'loaded' && node">
            <v-row justify="space-around">
                <v-checkbox v-model="turtle_mode" label="Turtle mode" />
                <v-col>
                    <v-tooltip top>
                        <template v-slot:activator="{ on }">
                            <v-btn
                                :disabled="saving"
                                color="success"
                                v-on="on"
                                @click.prevent="membersOk"
                            >
                                <v-icon dark>mdi-check-all</v-icon>OK
                            </v-btn>
                        </template>
                        <span>All visible recommendations match without exception. Increase left limit.</span>
                        <kbd>F</kbd>
                    </v-tooltip>
                </v-col>
                <v-col>
                    <v-tooltip top v-model="notOkTooltipModel">
                        <template v-slot:activator="{ on }">
                            <v-btn
                                id="button-not-ok"
                                :disabled="saving"
                                color="error"
                                v-on="on"
                                @click.prevent="membersNotOk"
                            >
                                <v-icon dark>mdi-close</v-icon>Not OK
                            </v-btn>
                        </template>
                        <span v-html="not_ok_tooltip" />
                    </v-tooltip>
                </v-col>
                <v-col>
                    <v-tooltip>
                        <template v-slot:activator="{ on }">
                            <v-btn
                                :disabled="!saved"
                                color="secondary"
                                v-on="on"
                                @click.prevent="initialize"
                            >
                                <v-icon dark>mdi-restart</v-icon>Start over
                            </v-btn>
                        </template>
                        Discard progress and start over.
                        <kbd>R</kbd>
                    </v-tooltip>
                </v-col>
                <v-col>
                    <v-tooltip>
                        <template v-slot:activator="{ on }">
                            <v-btn
                                :disabled="!saved"
                                color="secondary"
                                v-on="on"
                                @click.prevent="next"
                            >
                                <v-icon dark>mdi-chevron-right</v-icon>Next
                            </v-btn>
                        </template>
                        Continue with next node.
                        <kbd>N</kbd>
                    </v-tooltip>
                </v-col>
            </v-row>
        </div>
        <message-log class="bg-light" :messages="messages" />
        <v-dialog persistent v-model="doneDialogModel">
            <v-card>
                <v-card-title>
                    <span class="headline">Growing is done for this project.</span>
                </v-card-title>
                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn
                        v-if="project"
                        variant="primary"
                        :to="{
                        name: 'dataset',
                        params: { dataset_id: project.dataset_id }
                    }"
                    >Back to dataset</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>
    </div>
    <message-log
      class="bg-light"
      :messages="messages"
    />
    <b-modal
      ref="doneModal"
      centered
      no-fade
      header-bg-variant="success"
      title="Bisection done"
    >
      <div class="d-block text-center">
        Bisection is done for this project.
      </div>
      <footer slot="modal-footer">
        <b-button
          variant="primary"
          :to="{ name: 'projects' }"
        >Back to projects</b-button>
      </footer>
    </b-modal>
  </div>
</template>

<script>
import axios from "axios";
import shuffle from "lodash/shuffle";

import InfiniteLoading from "vue-infinite-loading";

import mixins from "@/mixins.js";
import * as api from "@/helpers/api.js";
import globalState from "@/globalState.js";
import exceptions from "@/exceptions.js";
import EventLog from "@/EventLog.js";

import MemberPreview from "@/components/MemberPreview.vue";
import MessageLog from "@/components/MessageLog.vue";

const MAX_N_RECOMMENDATIONS = 100000;

export default {
    name: "grow",
    data() {
        return {
            node_status: "",
            project: null,
            // true, if the project needs to be reloaded (e.g. when the progress needs to be updated).
            project_dirty: false,
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
            // Page that is actually currently shown
            rec_current_page_shown: null,
            rec_base_url: null,
            rec_n_pages: null,
            rec_request_id: null,
            rec_status: "",
            done: false,
            rec_member_controls: [
                {
                    event: "remove",
                    icon: "mdi-close",
                    title: "Remove this object from the suggestions."
                },
                {
                    event: "accept",
                    icon: "mdi-check",
                    title: "Accept this object."
                }
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

            // View models
            notOkTooltipModel: false,
            doneDialogModel: false,

            /* Accepted members */
            accepted_members: [],

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
                time_saved: null
            },

            // Log all the relevent events
            eventLog: new EventLog(),

            // Time when page was first entered
            pageEnteredTime: null
        };
    },
    components: {
        MemberPreview,
        InfiniteLoading,
        MessageLog
    },
    mixins: [mixins],
    watch: {
        turtle_mode: function(value) {
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

            this.notOkTooltipModel = true;
        }
    },
    mounted() {
        window.addEventListener("keypress", this.keypress);
    },
    beforeDestroy() {
        window.removeEventListener("keypress", this.keypress);
    },
    beforeRouteEnter(to, from, next) {
        next(vm => {
            vm.updateView(to.params.project_id, to.params.node_id);
        });
    },
    beforeRouteUpdate(to, from, next) {
        this.updateView(to.params.project_id, to.params.node_id);
        next();
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
            if (this.turtle_mode) {
                return "<strong>All</strong> visible recommendations <strong>do not match</strong> without exception. Save all as rejected and proceed. <kbd>J</kbd>";
            }
            return "<strong>Some</strong> visible recommendations do not match. Decrease right limit. <kbd>J</kbd>";
        }
    },
    methods: {
        updateView(project_id, node_id) {
            globalState.setLoading("growing");

            if (this.pageEnteredTime == null) {
                this.pageEnteredTime = Date.now();
            }

            // Load project if dirty
            const projectPromise =
                this.project && !this.project_dirty
                    ? Promise.resolve(this.project)
                    : (globalState.setLoading("project"),
                      api.getProject(project_id, true, true));
            projectPromise.then(project => {
                this.project = project;
                this.project_dirty = false;
                globalState.unsetLoading("project");
            });

            if (node_id == null) {
                console.log(`getNextUnfilledNode ${project_id}...`);
                api.getNextUnfilledNode(project_id, null, true, true)
                    .then(node_id => {
                        if (node_id == null) {
                            throw exceptions.NoNextNodeException(
                                `No ungrown nodes for project ${project_id}.`
                            );
                        }

                        this.$router.push({
                            name: "grow",
                            params: {
                                project_id: project_id,
                                node_id: node_id
                            }
                        });
                    })
                    .catch(e => {
                        console.log(e);
                        this.initializeView(null, e);
                    })
                    .finally(() => {
                        globalState.unsetLoading("growing");
                    });
                return;
            }

            const nodePromise = api.getNode(project_id, node_id);

            Promise.all([projectPromise, nodePromise])
                .then(([, node]) => {
                    console.log(node);
                    this.initializeView(node, null);
                })
                .catch(e => {
                    console.log(e);
                    this.initializeView(null, e);
                })
                .finally(() => {
                    globalState.unsetLoading("growing");
                });
        },
        initializeView(node, error) {
            // Completely reset data
            Object.assign(this.$data, this.$options.data(), {
                // project and project_dirty are initialized by updateView
                project: this.project,
                project_dirty: this.project_dirty,
                node,
                view_valid: !error
            });

            if (error instanceof exceptions.NoNextNodeException) {
                this.done = true;
            } else if (error) {
                this.error = error;
                return;
            }

            // Log pageEntered
            this.eventLog.logEvent("pageEntered", null, this.pageEnteredTime);
            this.pageEnteredTime = null;

            // Reset infinite-loading
            if (this.$refs.InfiniteLoading) {
                this.$refs.InfiniteLoading.stateChanger.reset();
            }

            // Update breadcrumb
            this.setBreadcrumbs([
                {
                    text: this.project.dataset.name,
                    to: {
                        name: "dataset",
                        params: { dataset_id: this.project.dataset.dataset_id }
                    },
                    exact: true
                },
                {
                    text: this.project.name,
                    to: {
                        name: "project",
                        params: { project_id: this.project.project_id }
                    },
                    exact: true
                },
                {
                    text: "Grow"
                },
                {
                    text: this.node.name,
                    to: {
                        name: "grow",
                        params: {
                            project_id: this.project.project_id,
                            node_id: this.node.node_id
                        }
                    }
                }
            ]);

            this.updateRecommendations();
        },
        updateRecommendations() {
            /**
             * Update the recommendation section.
             */

            if (!this.rec_base_url) {
                // if rec_base_url is not known
                api.getNodeRecommendedObjects(
                    this.project.project_id,
                    this.node.node_id,
                    MAX_N_RECOMMENDATIONS
                )
                    .then(data => {
                        // TODO: Do something when there are no recommendations!
                        this.rec_members = shuffle(data.data);
                        this.rec_base_url = data.links.self;
                        this.rec_n_pages = this.rec_interval_right =
                            data.meta.last_page + 1;

                        this.rec_current_page = this.rec_interval_left = 0;
                        this.rec_status = "loaded";
                        this.rec_request_id = data.meta.request_id;

                        this.eventLog.logEvent("recommendationsInitialized");
                    })
                    .catch(e => {
                        this.axiosErrorHandler(e);
                    });
            } else {
                // If rec_base_url is known
                axios
                    .get(`${this.rec_base_url}&page=${this.rec_current_page}`)
                    .then(response => {
                        console.log(
                            response.data.data,
                            shuffle(response.data.data)
                        );
                        this.rec_members = shuffle(response.data.data);
                        this.rec_current_page_shown = response.data.meta.page;

                        this.eventLog.logEvent("recommendationsUpdated", {
                            current_page: response.data.meta.page
                        });
                    })
                    .catch(e => {
                        this.axiosErrorHandler(e);
                    });
            }
        },
        initialize__() {
            console.log("Initializing...");

            // Reset data (but keep project)
            Object.assign(this.$data, this.$options.data(), {
                project: this.project
            });

            // Time when the view is visited
            this.log_data.time_visit = Date.now();

            const project_id = parseInt(this.$route.params.project_id);

            this.node_status = "loading";

            var projectPromise = new Promise(resolve => {
                if (this.project && this.project.project_id == project_id) {
                    // Project was already loaded.
                    resolve();
                } else {
                    // Load project
                    this.project = null;
                    api.getProject(project_id, true).then(project => {
                        this.project = project;
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
                        .getNextUnfilledNode(
                            this.project.project_id,
                            this.project.node_id,
                            true,
                            true
                        )
                        // (This really needs to be nested!)
                        .then(node_id => {
                            if (node_id === null) {
                                // Done
                                this.$refs.doneModal.show();
                                return Promise.reject(
                                    new Error("No next node")
                                );
                            }
                            const to = {
                                name: "grow",
                                params: {
                                    project_id: project_id,
                                    node_id: node_id
                                }
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
                .then(node_id => {
                    return api.getNode(project_id, node_id).then(node => {
                        this.node = node;
                    });
                })
                .then(() => {
                    this.node_status = "loaded";
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });

            nodeIdPromise
                .then(node_id => {
                    console.log("getNodeRecommendedObjects...");
                    this.rec_status = "loading";

                    return api.getNodeRecommendedObjects(
                        project_id,
                        node_id,
                        MAX_N_RECOMMENDATIONS
                    );
                })
                .then(data => {
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
                .catch(e => {
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
                this.node_members_url = `/api/projects/${
                    this.node.project_id
                }/nodes/${
                    this.node.node_id
                }/members?objects=${!nodes}&nodes=${nodes}&arrange_by=random&`;
                this.node_members_page = 0;
                updateMembersUrl = true;
            }

            axios
                .get(`${this.node_members_url}&page=${this.node_members_page}`)
                .then(response => {
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
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        membersOk: function() {
            // Increase umber of decisions
            this.log_data.n_accept_page++;

            this.rec_interval_left = this.rec_current_page + 1;

            this.accepted_members = [];

            this.updateCurrentPage();

            this.showNext();
        },
        membersNotOk: function() {
            // Increase umber of decisions
            this.log_data.n_reject_page++;

            if (this.accepted_members.length) {
                // If there are accepted members, reject all remaining and proceed like in membersOk
                var remaining_members = this.rec_members.map(this.getUniqueId);

                console.log("Rejecting", remaining_members);
                this.rejected_members.push(...remaining_members);

                this.rec_interval_left = this.rec_current_page + 1;
                this.accepted_members = [];
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
                // and increase leap.import InfiniteLoading from "vue-infinite-loading";
                this.rec_current_page = Math.min(
                    this.rec_current_page + this.jump_pages,
                    this.rec_n_pages - 1
                );
                this.jump_pages *= 2;
            } else {
                // Otherwise perform regular bisectionimport InfiniteLoading from "vue-infinite-loading";
                this.rec_current_page = Math.trunc(
                    (1 - frac) * this.rec_interval_left +
                        frac * this.rec_interval_right
                );
            }
        },
        showNext: function() {
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

            axios
                .get(`${this.rec_base_url}&page=${this.rec_current_page}`)
                .then(response => {
                    console.log(
                        response.data.data,
                        shuffle(response.data.data)
                    );
                    this.rec_members = shuffle(response.data.data);
                })
                .catch(e => {
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

            // TODO: Log viewed pages
            api.nodeAcceptRecommendations(
                node.project_id,
                node.node_id,
                this.rec_request_id,
                this.rejected_members,
                this.rec_interval_left - 1
            )
                .then(() => {
                    console.log("Saved all recommendations.");
                })
                .then(() => {
                    return api.patchNode(node.project_id, node.node_id, {
                        filled: true
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
                .catch(e => {
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

            this.accepted_members.push(this.getUniqueId(member));
            this.messages.unshift(
                `Accepted ${this.accepted_members.length} objects.`
            );
        },
        next() {
            this.$router.push({
                name: "grow",
                params: { project_id: this.project.project_id }
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
        }
    }
};
</script>

<style>
#grow {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
    height: 100%;
}

#grow > * {
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
