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
    string s, t;
    cin >> s >> t;
    int len = 0;
    if (t.length() > 2)
        len = 100;
    else
    {
        for (int i = 0; i < t.length(); i++)
        {
            int temp = t[i] - '0';
            for (int j = t.length() - 1 - i; j >= 1; j--)
                temp *= 10;
            len += temp;
        }
    }
    // cout << len << endl;
    auto leng = [&](string std) -> int
    {
        int l = std.length();
        vector<int> dp(l, 1);
        for (int i = 1; i < l; i++)
            for (int j = 0; j < i; j++)
                if (std[j] < std[i])
                    dp[i] = max(dp[i], dp[j] + 1);
        return *max_element(dp.begin(), dp.end());
    };
    set<char> st;
    map<char, int> mp;
    for (int i = 0; i < s.length(); i++)
    {
        st.insert(s[i]);
        mp[s[i]]++;
    }
    if (st.size() <= len)
    {
        cout << st.size() << endl;
        return;
    }
    else
    {
        string std = "";
        for (int i = 1; i <= len; i++)
            std += s;
        cout << leng(std) << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}