<template>
    <div id="bisect">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark">
            <router-link class="navbar-brand text-light" to="/">MorphoCluster</router-link>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav mr-auto">
                    <li class="nav-item nav-link text-light" v-if="project">
                        {{project.name}}
                    </li>
                    <li class="nav-item nav-link  text-light">
                        Bisect
                    </li>
                    <li class="nav-item  nav-link text-light" v-if="node">
                        {{node.name}}
                    </li>
                </ul>
            </div>
        </nav>
        <div v-if="loading">Loading...</div>
        <div class="bg-light section-heading border-bottom border-top">Node members
            <span v-if="node">({{node.n_objects_deep}})</span>
        </div>
        <div id="node-members" class="row scrollable">
            <div v-if="node" class="col col-1">
                <member-preview v-bind:member="node" />
            </div>

            <div :key="getUniqueId(m)" v-for="m of node_members" class="col col-1">
                <member-preview v-bind:member="m" />
            </div>

            <infinite-loading ref="infload" v-if="node" @infinite="updateNodeMembers" spinner="circles">
                <div slot="no-more" />
            </infinite-loading>
        </div>
        <div v-if="recommended_members && !done" class="bg-light section-heading border-bottom border-top">Recommended members
            <span v-if="typeof(current_page) != 'undefined'">(Page {{current_page + 1}})</span>
        </div>
        <div id="recommended-members" v-if="recommended_members && !done" class="row scrollable">
            <div :key="getUniqueId(m)" v-for="m of recommended_members" class="col col-1">
                <member-preview :member="m" :controls="rec_member_controls" v-on:remove="removeMember" />
            </div>
        </div>
        <div v-if="done" class="bg-light section-heading">Report</div>
        <div id="report" v-if="done" class="scrollable">
            Bisection done.
            <table>
                <tr>
                    <th>Total number of pages:</th>
                    <td>{{n_pages}}</td>
                </tr>
                <tr>
                    <th>Number of valid pages:</th>
                    <td>{{n_valid_pages}}</td>
                </tr>
                <tr>
                    <th>Number of invalid pages:</th>
                    <td>{{n_invalid_pages}}</td>
                </tr>
                <tr>
                    <th>Number of rejected members:</th>
                    <td>{{rejected_members.length}}</td>
                </tr>
            </table>
            <p v-if="n_valid_pages == n_pages">
                You accepted all recommendations. You may want to
                <i>start over</i> to get more.
            </p>
        </div>
        <div id="progress">
            <div :style="{flexGrow: n_valid_pages}" class="bg-success" />
            <div :style="{flexGrow: n_unsure_pages}" class="bg-warning" />
            <div :style="{flexGrow: n_invalid_pages}" class="bg-danger" />
        </div>
        <div id="decision" v-if="node">
            <b-button variant="success" v-b-tooltip.hover.html title="All recommended members match without exception. Increase left limit. <kbd>F</kbd>" @click.prevent="membersOk">
                <i class="mdi mdi-check" /> OK</b-button>
            <b-button variant="danger" v-b-tooltip.hover.html title="Some recommended members do not match. Decrease right limit. <kbd>J</kbd>" @click.prevent="membersNotOk">
                <i class="mdi mdi-close" /> Not OK</b-button>
            <b-button variant="secondary" v-b-tooltip.hover.html title="Discard progress and start over. <kbd>R</kbd>" @click.prevent="initialize">
                <i class="mdi mdi-restart" /> Start over</b-button>
            <!-- <b-button variant="outline-success" v-b-tooltip.hover title="Assign all safe objects to the current node." @click.prevent="saveResult">Save result</b-button> -->
            <!-- <div>
        n_valid_pages: {{n_valid_pages}}, n_unsure_pages: {{n_unsure_pages}}, n_invalid_pages: {{n_invalid_pages}}, interval_left: {{interval_left}}, interval_right: {{interval_right}}
      </div> -->
            <b-button :disabled="!done" variant="secondary" v-b-tooltip.hover.html title="Continue with next node. <kbd>N</kbd>" @click.prevent="next">
                <i class="mdi mdi-chevron-right" /> Next
            </b-button>
        </div>
        <message-log class="bg-light" :messages="messages" />
    </div>
</template>

<script>
import axios from "axios";
import shuffle from "lodash/shuffle";

import InfiniteLoading from "vue-infinite-loading";

import mixins from "@/mixins.js";

import * as api from "@/helpers/api.js";

import MemberPreview from "@/components/MemberPreview.vue";
import MessageLog from "@/components/MessageLog.vue";

export default {
    name: "bisect",
    data() {
        return {
            loading: null,
            project: null,
            node: null,
            node_members: [],
            members_url: null,
            recommended_members: [],
            rejected_members: [],
            interval_left: 0,
            interval_right: null,
            current_page: 0,
            base_url: null,
            n_pages: null,
            done: false,
            rec_member_controls: [
                {
                    event: "remove",
                    icon: "mdi-close",
                    title: "Remove this member from the suggestions."
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
            jump_pages: 1
        };
    },
    components: {
        MemberPreview,
        InfiniteLoading,
        MessageLog
    },
    mixins: [mixins],
    watch: {
        $route: "initialize"
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
            return this.interval_right - this.n_unsure_pages;
        },
        n_unsure_pages() {
            return Math.max(0, this.interval_right - this.interval_left);
        },
        n_invalid_pages() {
            return this.n_pages - this.interval_right;
        }
    },
    methods: {
        initialize() {
            // Reset data (but keep project)
            Object.assign(this.$data, this.$options.data(), {
                project: this.project
            });

            const project_id = parseInt(this.$route.params.project_id);

            var projectPromise = new Promise(resolve => {
                if (this.project && this.project.project_id == project_id) {
                    // Project was already loaded.
                    resolve();
                } else {
                    // Load project
                    this.project = null;
                    this.progress = null;
                    api.getProject(project_id, true).then(project => {
                        this.project = project;
                        this.progress = {
                            n_approved_objects: project.n_approved_objects,
                            n_objects_total: project.n_objects_total
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
                return api
                    .getNextUnfilledNode(this.project.node_id, true)
                    .then(node_id => {
                        if (node_id === null) {
                            // Done
                            return Promise.reject(new Error("No next node"));
                        }
                        const to = {
                            name: "bisect",
                            params: {
                                project_id: project_id,
                                node_id: node_id
                            }
                        };

                        this.$router.replace(to);
                        return node_id;
                    });
            });

            nodeIdPromise
                .then(node_id => {
                    return api.getNode(node_id).then(node => {
                        this.node = node;
                    });
                })
                .then(() => {
                    this.loading = false;
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });

            nodeIdPromise.then(node_id => {
                api.getNodeRecommendedObjects(node_id, 5000).then(data => {
                    this.recommended_members = shuffle(data.data);
                    this.base_url = data.links.self;
                    this.n_pages = this.interval_right =
                        data.meta.last_page + 1;

                    this.current_page = this.interval_left = 0;
                });
            });
        },
        // updateNodeMembers gets called as an infinite loading handler.
        updateNodeMembers($state) {
            if (!this.node) {
                $state.reset();
                return;
            }
            console.log("updateNodeMembers");

            // Should members_url be updated (with unique id etc.) on response?
            var updateMembersUrl = false;

            if (!this.members_url) {
                const nodes = !!this.node.children;
                this.members_url = `/api/nodes/${
                    this.node.node_id
                }/members?objects=${!nodes}&nodes=${nodes}&arrange_by=interleaved&`;
                this.page = 0;
                updateMembersUrl = true;
            }

            axios
                .get(`${this.members_url}&page=${this.page}`)
                .then(response => {
                    this.node_members = this.node_members.concat(
                        response.data.data
                    );

                    if (updateMembersUrl) {
                        this.members_url = response.data.links.self;
                    }

                    $state.loaded();

                    if (this.page < response.data.meta.n_pages) {
                        this.page += 1;
                    } else {
                        $state.complete();
                    }
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        membersOk: function() {
            this.interval_left = this.current_page + 1;

            if (!this.found_right) {
                this.current_page += this.jump_pages;
                this.jump_pages *= 2;
            } else {
                this.updateCurrentPage();
            }

            this.showNext();
        },
        membersNotOk: function() {
            this.interval_right = this.current_page;
            this.found_right = true;

            this.updateCurrentPage(0.25);

            this.showNext();
        },
        updateCurrentPage(frac = 0.5) {
            this.current_page = Math.trunc(
                (1 - frac) * this.interval_left + frac * this.interval_right
            );
        },
        showNext: function() {
            console.log(
                this.current_page,
                this.interval_left,
                this.interval_right
            );

            if (this.n_unsure_pages == 0) {
                this.done = true;

                this.saveResult();
                return;
            }

            axios
                .get(`${this.base_url}&page=${this.current_page}`)
                .then(response => {
                    console.log(
                        response.data.data,
                        shuffle(response.data.data)
                    );
                    this.recommended_members = shuffle(response.data.data);
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        saveResult() {
            // For each page in the valid range (0<=x<this.interval_left):
            // Fetch members and assign to the current node.

            var promises = Array(this.interval_left)
                .fill()
                .map((v, i) => {
                    return axios
                        .get(`${this.base_url}&page=${i}`)
                        .then(response => {
                            var members = response.data.data.filter(m => {
                                return !this.rejected_members.includes(
                                    this.getUniqueId(m)
                                );
                            });

                            return axios.post(
                                `/api/nodes/${this.node.node_id}/adopt_members`,
                                {
                                    members
                                }
                            );
                        });
                });

            promises.push(api.patchNode(this.node.node_id, { filled: true }));

            Promise.all(promises)
                .then(() => {})
                .catch(e => {
                    this.messages.unshift("Error");
                    console.log(e);
                });
        },
        removeMember(member) {
            console.log("Remove", this.getUniqueId(member));

            // Remove from current recommendations
            var index = this.recommended_members.indexOf(member);
            if (index > -1) {
                this.recommended_members.splice(index, 1);
            }

            // And add to rejected
            this.rejected_members.push(this.getUniqueId(member));
        },
        next() {
            this.$router.push({
                name: "bisect",
                params: { project_id: this.project.project_id }
            });
        },
        keypress(event) {
            if (
                this.loading ||
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
</style>
