#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
#define yes cout << "YES" << endl
#define no cout << "NO" << endl
#define yess cout << "Yes" << endl
#define noo cout << "No" << endl
using namespace std;
vector<int> pre;
vector<int> ans;
int cnt;
int k;
void pas()
{
    auto it = upper_bound(pre.begin(), pre.end(), k) - pre.begin() - 1;
    k -= pre[it];
    //cout << pre[it] << " " << k << " " << it << endl;
    ans.pb(it);
    cnt += it + 1;
}
void solve()
{
    ans.clear();
    cin >> k;
    cnt = 0;
    while (k)
    {
        pas();
    }
    cout << cnt << endl;
    int temp = 1;
    for (auto i : ans)
    {
        for (int j = 0; j < i + 1; j++)
        {
            cout << temp << " " << temp + j << endl;
        }
        temp = temp + i + 1;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    pre.pb(0);
    for (int t = 1; (t * (t + 1)) / 2 <= 1e6; t++)
        pre.pb(t * (t + 1) / 2);
    while (T--)
    {
        solve();
    }
    return 0;
}