#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n;
    cin >> n;
    string s;
    cin >> s;
    int cnt0 = 0, cnt1 = 0;
    for (auto c : s)
    {
        if (c == '0')
            cnt0++;
        else
            cnt1++;
    }
    int ans = 0;
    auto check = [=](int c0, int c1, int cc)
    {
        for (int i = s.size() - 1; i >= cc; i--)
        {
            if (c0 == 0 && c1 == 0)
                break;
            if (c0 % 2 == 0 && c1 % 2 == 0)
                return true;
            if (s[i] == '0')
                c0--;
            else
                c1--;
        }
        return false;
    };
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] == '0')
            cnt0--;
        else
            cnt1--;
        if (check(cnt0, cnt1, i + 1))
            ans++;
    }
    cout << (double)ans / n << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}