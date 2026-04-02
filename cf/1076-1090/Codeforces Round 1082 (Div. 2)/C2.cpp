#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef long long ll;

// 树状数组用于维护区间内的最大位置
struct Fenwick
{
    int n;
    vector<int> tree;
    Fenwick(int n) : n(n), tree(n + 1, 0) {}

    void update(int i, int val)
    {
        for (; i <= n; i += i & -i)
        {
            tree[i] = max(tree[i], val);
        }
    }

    int query(int i)
    {
        int res = 0;
        for (; i > 0; i -= i & -i)
        {
            res = max(res, tree[i]);
        }
        return res;
    }
};

void solve()
{
    int n;
    if (!(cin >> n))
        return;
    vector<int> a(n + 1);
    vector<int> vals;
    for (int i = 1; i <= n; ++i)
    {
        cin >> a[i];
        vals.push_back(a[i]);
    }

    // 坐标压缩
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = vals.size();

    auto get_rank = [&](int v)
    {
        return lower_bound(vals.begin(), vals.end(), v) - vals.begin() + 1;
    };

    Fenwick ft(m);
    vector<int> last_pos(m + 1, 0);
    ll total_sum = 0;

    for (int i = 1; i <= n; ++i)
    {
        int cur_val = a[i];
        int r_cur = get_rank(cur_val);

        // 找到最近的 a[i]-1 的位置 L
        int L = 0;
        auto it = lower_bound(vals.begin(), vals.end(), cur_val - 1);
        if (it != vals.end() && *it == cur_val - 1)
        {
            L = last_pos[it - vals.begin() + 1];
        }

        // 找到最近的小于 a[i]-1 的位置 ns
        // 使用 lower_bound 找到第一个 >= a[i]-1 的位置，其索引即为小于 a[i]-1 的元素个数
        int limit_idx = lower_bound(vals.begin(), vals.end(), cur_val - 1) - vals.begin();
        int ns = ft.query(limit_idx);

        // 如果 L 被更小的数阻隔，则无法覆盖
        int j_star = (L > ns) ? L : 0;

        // 计算贡献：(i - j_star) 是有效的左端点数量，(n - i + 1) 是有效的右端点数量
        total_sum += (ll)(i - j_star) * (n - i + 1);

        // 更新维护信息
        last_pos[r_cur] = i;
        ft.update(r_cur, i);
    }

    cout << total_sum << "\n";
}

int main()
{
    // 优化输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}