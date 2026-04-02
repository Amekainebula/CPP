#include <bits/stdc++.h>
using namespace std;
#define int long long

void solve()
{
    map<int, vector<int>> pos;
    map<int, int> ptr;
    int n, q;
    cin >> n >> q;
    int arr[n + 1];
    for (int i = 1; i <= n; i++)
    {
        cin >> arr[i];
        pos[arr[i]].push_back(i);
        ptr[arr[i]] = 0;
    }
    vector<tuple<int, int, int, int>> qr(q);
    int c = 0;
    for (auto &[l, r, v, i] : qr)
    {
        cin >> v >> l >> r;
        i = c++;
    }
    vector<int> anss(q);
    sort(qr.begin(), qr.end());
    int prevL = 1;
    for (auto [l, r, v, idd] : qr)
    {
        for (int j = prevL; j < l; j++)
        {
            ptr[arr[j]]++;
        }
        prevL = l;
        vector<pair<int, int>> facts;
        for (int i = 1; i * i <= v; i++)
        {
            if (v % i == 0)
            {
                int fact1 = v / i;
                if (!(pos[fact1].size() == 0 || ptr[fact1] >= pos[fact1].size() ||
                 pos[fact1][ptr[fact1]] > r || fact1 == 1))
                {
                    facts.push_back({pos[fact1][ptr[fact1]], fact1});
                }
                int fact2 = i;
                if (fact2 == fact1)
                    continue;
                if (!(pos[fact2].size() == 0 || ptr[fact2] >= pos[fact2].size() || 
                pos[fact2][ptr[fact2]] > r || fact2 == 1))
                {
                    facts.push_back({pos[fact2][ptr[fact2]], fact2});
                }
            }
        }
        sort(facts.begin(), facts.end());
        int pr = l;
        int ans = 0;
        for (auto [idx, val] : facts)
        {
            int rr = idx - pr;
            ans += v * rr;
            while (v % val == 0)
                v /= val;
            pr = idx;
        }
        if (facts.empty())
            ans = v * (r - l + 1);
        else
            ans += (r - pr + 1) * v;
        anss[idd] = ans;
    }
    for (int i = 0; i < q; i++)
        cout << anss[i] << "\n";
}

signed main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int t;
    cin >> t;
    while (t--)
        solve();
    return 0;
}