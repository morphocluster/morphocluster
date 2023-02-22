<template>
    <div id="labeling2">
        <h2 v-if="this.node">{{this.node.name}}</h2>
        <div id="node-info" class="scrollable">
            <node-header v-if="node" :node="node"/>

            <div class="row">
                <div v-for="m in node_members" class="col col-1" :key="getUniqueId(m)">
                    <member-preview
                        v-bind:member="m"
                        />
                </div>
            </div>
            <infinite-loading
                @infinite="updateMembers"
                spinner="circles" />
        </div>
    </div>
</template>

<script>
import axios from "axios";
import InfiniteLoading from "vue-infinite-loading";

import MemberPreview from "@/components/MemberPreview.vue";
import NodeHeader from "@/components/NodeHeader.vue";

export default {
  name: "Labeling2View",
  props: {
    node_id: Number
  },
  components: {
    MemberPreview,
    InfiniteLoading,
    NodeHeader
  },
  data() {
    return {
      node: null,
      node_members: [],
      message: null,
      members_url: null,
      page: null
    };
  },
  methods: {
    setMessage: function(msg) {
      this.message = msg;
    },
    updateMembers($state) {
      var updateMembersUrl = false;

      if (!this.members_url) {
        this.members_url = `/api/nodes/${
          this.node_id
        }/members?objects=true&arrange_by=interleaved`;
        this.page = 0;
        updateMembersUrl = true;
      } else {
        this.page += 1;
      }

      var url = `${this.members_url}&page=${this.page}`;

      console.log(`Loading ${url}...`);

      axios
        .get(`${this.members_url}&page=${this.page}`)
        .then(response => {
          this.node_members = this.node_members.concat(response.data.data);
          if (updateMembersUrl) {
            this.members_url = response.data.links.self;
            console.log(this.members_url);
          }

          $state.loaded();
          console.log("Done loading.");
        })
        .catch(e => {
          this.setMessage(e.message);
          console.log(e);
        });
    }
  },
  mounted() {
    // Load node info
    axios
      .get(`/api/nodes/${this.node_id}`)
      .then(response => {
        this.node = response.data;
      })
      .catch(e => {
        this.setMessage(e.message);
        console.log(e);
      });
  }
};
</script>

<style>
#labeling2 {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  flex: 1;
  overflow: hidden;
}

#labeling2 > * {
  padding: 0 10px;
}

.scrollable {
  margin: 0;
  overflow-y: auto;
}

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
